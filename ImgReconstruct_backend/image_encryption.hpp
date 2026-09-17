#ifndef ENCRYPT_IMAGE_HPP
#define ENCRYPT_IMAGE_HPP

#include "CS_encryption.hpp"

class encrypt_image : CSencryption
{
private:
    float bm = 0.0f;

public:
    encrypt_image(std::string input_path);

    encrypt_image(const cv::Mat& input, bool global_image);

    encrypt_image() {}

    void get_mat(cv::Mat& dest);

    cv::Mat get_mat();

    void encrypt(const float& pixel_p, const std::string& password);

    void encrypt(const std::vector<int>& ri_x_g, const std::vector<int>& ri_y_g);

    void writeEncryptedImageToDisk(const std::string& output_path);

    cv::Size get_size(); 

    /** @brief encrypts an image; pass output_path = "" to skip writing to disk
    @param encrypted_out : if non-null, receives the encrypted image in memory (no PNG roundtrip)
    */
    static int encrypt_image_tiled(const std::string& input_path, const std::string& output_path, const std::string& password, float compression_ratio = 1.0f, cv::Mat* encrypted_out = nullptr);

private:
};

/** @brief encrypts a given image and writes the result to disk
@param input_path : path to input image
@param output_path : path to output image
@param password : password to be used for encryption and decryption
@param compression_ratio : compression ratio given as a value from 0 to 1, lower value means higher compression
*/

#endif