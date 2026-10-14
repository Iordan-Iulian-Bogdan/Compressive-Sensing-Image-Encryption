// Controlled single-image measurement-shape experiment. No encryption/container
// changes: all operators use the same CPU DCT-FISTA implementation below.
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <chrono>

struct Measurement { std::array<int, 4> p{}; int n = 0; };
using Operator = std::vector<Measurement>;

static Operator make_operator(int mode, int side, int ox, int oy, int seed,
    const cv::Mat& budgets) {
    Operator op;
    // Four disjoint T tetrominoes, each containing four pixels.
    constexpr int t[16] = {0,0,0,2, 1,0,2,2, 1,1,3,2, 1,3,3,3};
    for (int y = 0; y < side; y += 4) for (int x = 0; x < side; x += 4) {
        const int cy = (oy + y) / 4, cx = (ox + x) / 4;
        std::mt19937 rng((uint32_t)seed ^ (uint32_t)cx * 73856093u ^ (uint32_t)cy * 19349663u);
        std::array<int, 16> positions;
        std::iota(positions.begin(), positions.end(), 0);
        std::shuffle(positions.begin(), positions.end(), rng);
        std::array<int, 4> order{0,1,2,3};
        std::shuffle(order.begin(), order.end(), rng);
        const int count = budgets.at<uint8_t>(cy, cx);
        for (int k = 0; k < count; ++k) {
            Measurement m;
            for (int q = 0; q < 16; ++q) {
                const bool selected = mode == 0 ? q == positions[k] :
                    (mode == 1 ? (q / 8) * 2 + (q % 4) / 2 == order[k] : t[q] == order[k]);
                if (selected) m.p[m.n++] = (y + q / 4) * side + x + q % 4;
            }
            op.push_back(m);
        }
    }
    return op;
}

// Row-normalized operators A = mean * sqrt(group size). Each disjoint row
// has unit norm, so ||A||=1 for all shapes and the same FISTA step is valid.
static std::vector<float> forward(const cv::Mat& x, const Operator& op) {
    std::vector<float> b(op.size());
    const float* p = x.ptr<float>();
    for (size_t k = 0; k < op.size(); ++k) {
        for (int j = 0; j < op[k].n; ++j) b[k] += p[op[k].p[j]];
        b[k] /= std::sqrt((float)op[k].n);
    }
    return b;
}
static cv::Mat adjoint(const std::vector<float>& b, const Operator& op, int side) {
    cv::Mat x = cv::Mat::zeros(side, side, CV_32F);
    for (size_t k = 0; k < op.size(); ++k)
        for (int j = 0; j < op[k].n; ++j)
            x.ptr<float>()[op[k].p[j]] += b[k] / std::sqrt((float)op[k].n);
    return x;
}
static cv::Mat solve(const cv::Mat& src, const Operator& op, int steps, float lambda) {
    std::vector<float> b(op.size());
    // Stored means: one unsigned byte per measurement, for every operator.
    for (size_t k = 0; k < b.size(); ++k) {
        const float scale = std::sqrt((float)op[k].n);
        int sum = 0;
        for (int j = 0; j < op[k].n; ++j)
            sum += std::clamp((int)std::lround(src.ptr<float>()[op[k].p[j]] * 255), 0, 255);
        const int byte = (sum + op[k].n / 2) / op[k].n;
        b[k] = byte / 255.0f * scale;
    }
        const int side = src.rows;
    cv::Mat x = cv::Mat::zeros(src.size(), CV_32F), z = x.clone();
    cv::Mat weights(src.size(), CV_32F, cv::Scalar(1));
    weights.at<float>(0,0) = 0; // DC remains unpenalized.
    for (int rw = 0; rw < 2; ++rw) {
        float momentum = 1;
        z = x.clone();
        for (int it = 0; it < steps; ++it) {
            cv::Mat pixels;
            cv::idct(z, pixels);
            auto residual = forward(pixels, op);
            for (size_t k = 0; k < b.size(); ++k) residual[k] -= b[k];
            cv::Mat grad = adjoint(residual, op, side);
            cv::dct(grad, grad);
            // min .5 ||A D^-1 x - b||^2 + lambda * ||W x||_1.
            cv::Mat next = z - grad;
            for (int k = 0; k < side * side; ++k) {
                const float v = next.ptr<float>()[k];
                next.ptr<float>()[k] = std::copysign(std::max(0.0f,
                    std::abs(v) - lambda * weights.ptr<float>()[k]), v);
            }
            const float mt = (1 + std::sqrt(1 + 4 * momentum * momentum)) / 2;
            z = next + ((momentum - 1) / mt) * (next - x);
            x = next;
            momentum = mt;
        }
        if (rw == 0) for (int k = 1; k < side * side; ++k)
            weights.ptr<float>()[k] = std::clamp(0.01f / (std::abs(x.ptr<float>()[k]) + 0.01f), 0.05f, 1.0f);
    }
    cv::Mat pixels;
    cv::idct(x, pixels);
    return pixels;
}

static void checks() {
    cv::Mat budgets(4, 4, CV_8U, cv::Scalar(4));
    cv::Mat x(16,16,CV_32F); cv::randu(x, -1, 1);
    for (int mode = 0; mode < 3; ++mode) {
        auto op = make_operator(mode,16,0,0,17,budgets);
        if (op.size() != 64) throw std::runtime_error("wrong measurement count");
        std::vector<int> uses(256);
        for (auto& m : op) for (int j = 0; j < m.n; ++j) ++uses[m.p[j]];
        for (int n : uses) if (n > 1 || (mode != 0 && n != 1)) throw std::runtime_error("overlapping/incomplete groups");
        if (mode == 2) for (const auto& m : op) {
            int leaves=0, branches=0;
            for(int j=0;j<m.n;++j) {
                int degree=0;
                for(int k=0;k<m.n;++k) {
                    const int dy=std::abs(m.p[j]/16-m.p[k]/16);
                    const int dx=std::abs(m.p[j]%16-m.p[k]%16);
                    if(dx+dy==1) ++degree;
                }
                leaves+=degree==1; branches+=degree==3;
            }
            if(leaves!=3 || branches!=1) throw std::runtime_error("group is not a T tetromino");
        }
        std::vector<float> y(op.size()); for (size_t k=0;k<y.size();++k) y[k]=std::sin((float)k);
        const auto ax=forward(x,op); double lhs=0;
        for(size_t k=0;k<y.size();++k) lhs+=ax[k]*y[k];
        const double rhs=x.dot(adjoint(y,op,16));
        if(std::abs(lhs-rhs)>1e-4) throw std::runtime_error("adjoint test failed");
        const auto aat=forward(adjoint(y,op,16),op);
        for(size_t k=0;k<y.size();++k)
            if(std::abs(aat[k]-y[k])>1e-6) throw std::runtime_error("unit row norm test failed");
        cv::Mat flat(16,16,CV_32F,cv::Scalar(128.0/255));
        if(cv::norm(solve(flat,op,80,0.005f),flat,cv::NORM_INF)>0.002)
            throw std::runtime_error("constant reconstruction failed");
    }
        std::cout << "PASS: T geometry, group partition, counts, adjoint identity, unit row norms, constant reconstruction\n";
}

int main(int argc, char** argv) {
    try {
        checks();
        if (argc == 2 && std::string(argv[1]) == "--self-test") return 0;
        if (argc < 3) throw std::runtime_error("usage: input.png output_prefix [steps=80] [lambda=.005] [seed=17] [seg_mask] [rate_scale=1]");
        const int steps = argc>3 ? std::stoi(argv[3]) : 80;
        const float lambda = argc>4 ? std::stof(argv[4]) : .005f;
        const int seed = argc>5 ? std::stoi(argv[5]) : 17;
        const double rate_scale = argc>7 ? std::stod(argv[7]) : 1;
        if(steps<1 || lambda<0 || rate_scale<=0 || rate_scale>1) throw std::runtime_error("invalid solver/budget settings");
        cv::Mat original=cv::imread(argv[1]);
        if(original.empty()) throw std::runtime_error("input image missing");
        constexpr int side=64, stride=48;
        const int w=((std::max)(side,original.cols)-side+stride-1)/stride*stride+side;
        const int h=((std::max)(side,original.rows)-side+stride-1)/stride*stride+side;
        cv::Mat padded; cv::copyMakeBorder(original,padded,0,h-original.rows,0,w-original.cols,cv::BORDER_REPLICATE);
        cv::Mat budgets(h/4,w/4,CV_8U,cv::Scalar(4));
        if(argc>6) {
            cv::Mat mask=cv::imread(argv[6]);
            if(mask.empty()) throw std::runtime_error("segmentation mask missing");
            cv::resize(mask,mask,original.size(),0,0,cv::INTER_NEAREST);
            cv::copyMakeBorder(mask,mask,0,h-original.rows,0,w-original.cols,cv::BORDER_REPLICATE);
            const cv::Vec3b palette[8]={{110,110,110},{70,200,70},{230,120,40},{100,225,245},{50,110,180},{210,70,200},{235,220,80},{50,170,245}};
            // Relative scene priorities, normalized so no cell exceeds the
            // tested four measurements. All shapes share exactly this schedule.
            const double rates[8]={.2,.825,.2,.2,.5,.8,.5,.5};
            double carry=0;
            for(int y=0;y<h;y+=4) for(int x=0;x<w;x+=4) {
                double target=0;
                for(int dy=0;dy<4;++dy) for(int dx=0;dx<4;++dx) {
                    auto color=mask.at<cv::Vec3b>(y+dy,x+dx); int best=0,dist=1000000;
                    for(int c=0;c<8;++c) {auto d=cv::Vec3i(color)-cv::Vec3i(palette[c]); int dd=d.dot(d);if(dd<dist){dist=dd;best=c;}}
                    target+=rates[best]/.825*rate_scale/4;
                }
                const double v=target+carry;
                const int n=std::clamp((int)std::floor(v+.5),0,4);
                carry=v-n; budgets.at<uint8_t>(y/4,x/4)=(uint8_t)n;
            }
        }
        const size_t measurements=(size_t)cv::sum(budgets)[0];
        std::ofstream csv(std::string(argv[2])+".csv");
        csv<<"operator,seed,steps_per_reweight,reweights,lambda,measurements_per_channel,payload_bytes,seconds,psnr,mae\n";
        cv::Mat float_image;padded.convertTo(float_image,CV_32FC3,1.0/255);
        std::vector<cv::Mat> planes;cv::split(float_image,planes);
        cv::Mat feather(side,side,CV_32F);
        for(int y=0;y<side;++y) for(int x=0;x<side;++x)
            feather.at<float>(y,x)=(float)(std::min({x+1,y+1,side-x,side-y,stride/2})/(double)(stride/2));
        const char* names[]={"random_points","square_means","tetromino_means"};
        for(int mode=0;mode<3;++mode) {
            const auto start=std::chrono::steady_clock::now();
            cv::Mat sum=cv::Mat::zeros(h,w,CV_32FC3),weight=cv::Mat::zeros(h,w,CV_32F);
            // Quantized bytes in global cell order; overlap reuses the same
            // measurements, not an additional acquisition or storage budget.
            std::ofstream payload(std::string(argv[2])+"."+names[mode]+".measurements.bin",std::ios::binary);
            for(int y=0;y<h;y+=4) for(int x=0;x<w;x+=4) {
                const auto cell=make_operator(mode,4,x,y,seed,budgets);
                for(const auto& m:cell) for(int c=0;c<3;++c) {
                    double mean=0;
                    for(int j=0;j<m.n;++j) mean+=padded.at<cv::Vec3b>(y+m.p[j]/4,x+m.p[j]%4)[c];
                    const uint8_t byte=(uint8_t)std::lround(mean/m.n);
                    payload.put((char)byte);
                }
            }
            payload.close();
            for(int y=0;y<=h-side;y+=stride) for(int x=0;x<=w-side;x+=stride) {
                cv::Rect r(x,y,side,side);auto op=make_operator(mode,side,x,y,seed,budgets);
                std::vector<cv::Mat> rec(3);
                for(int c=0;c<3;++c) {auto patch=planes[c](r).clone();rec[c]=solve(patch,op,steps,lambda);}
                cv::Mat merged,w3;cv::merge(rec,merged);cv::merge(std::vector<cv::Mat>{feather,feather,feather},w3);
                cv::multiply(merged,w3,merged);sum(r)+=merged;weight(r)+=feather;
            }
            std::vector<cv::Mat> rec;cv::split(sum,rec);for(auto& c:rec) cv::divide(c,weight,c);
            cv::Mat result;cv::merge(rec,result);result=result(cv::Rect(0,0,original.cols,original.rows)).clone();
            result.convertTo(result,CV_8UC3,255);
            const double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            cv::Mat diff;cv::absdiff(result,original,diff);const auto mae=cv::mean(diff);
            const double psnr=cv::PSNR(result,original);
            cv::imwrite(std::string(argv[2])+"."+names[mode]+".png",result);
            csv<<names[mode]<<','<<seed<<','<<steps<<",2,"<<lambda<<','<<measurements<<','<<measurements*3<<','<<secs<<','<<psnr<<','<<(mae[0]+mae[1]+mae[2])/3<<'\n';
            std::cout<<names[mode]<<" samples/channel="<<measurements<<" bytes="<<measurements*3<<" seconds="<<secs<<" PSNR="<<psnr<<" MAE="<<(mae[0]+mae[1]+mae[2])/3<<'\n';
        }
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
