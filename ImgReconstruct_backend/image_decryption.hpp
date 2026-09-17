#ifndef DECRYPTION_IMAGE_HPP
#define DECRYPTION_IMAGE_HPP

#include "CS_encryption.hpp"

class decrypt_image : CSencryption {
private:
    /** @brief shared header parsing + MAC verification for both constructors.
    Throws std::runtime_error on legacy containers, undecryptable headers or
    failed authentication (v2). On success fills m/rows/cols/org_size and,
    for v2 containers, caches the derived key for returnRandomIndices.
    */
    void parse_container_header(const cv::Mat& input, const std::string& password);

public:
    decrypt_image(const std::string input_path, const std::string& password = "");

    decrypt_image() {}

    decrypt_image(cv::Mat input, const std::string& password = "");

    void decrypt(cv::Mat ref[3], const std::vector<int>& ri_x_g, const std::vector<int>& ri_y_g, const int num_iterations, const float coef, cv::Mat& out, bool ycrcb = false);

    void get_mat(cv::Mat& dest);

    cv::Mat get_mat();

    void writeDecryptedImageToDisk(const std::string output_path, bool remove_noise = false, int noise_level = 3);

    /** @brief extracts the sampled pixels and their coordinates. Requires the
    derived key cached by the constructor (v2 containers authenticated with
    the password); index regeneration uses the cached 256-bit key.
    */
    void get_sampled_mat(cv::Mat& sampled_mat, cv::Mat& masked_mat);

    float get_compression_ratio();

    cv::Size get_org_size();

    /** @brief decrypts a given image and writes the result to disk
    @param input_path:  path to input image
    @param output_path:  path to output image
    @param password:  password to be used for encryption and decryption
    @param parameters_type: use AUTO_PARAM for automatically choosing parameters or MANUAL_PARAM for fine tuning
    @param num_tiles : number of tiles to be processed
    @param overlap : how many pixels should the tiles overlap, bigger number may result in better quality
    @param iterations : number of iterations, higher value may result in better quality
    @param nun_threads : number of CPU threads
    @param coef : coeficient used for the solver, higher value should be used for a more compressed image
    */
    /** @brief decrypts tiles from an already-loaded encrypted image (in-memory, no PNG roundtrip).
    Shared implementation for both overloads. Note: the Mat is const but the impl needs a copy.
    */
    static int decrypt_image_tiled(const cv::Mat& encrypted_img_in, const std::string& output_path, const std::string& password, int num_tiles = 24, int overlap = 48, int iterations = 20, int nun_threads = 8, float coef = 0.01f, bool show_preview = true, bool denoise = false);

    static int decrypt_image_tiled(const std::string& input_path, const std::string& output_path, const std::string& password, int num_tiles = 24, int overlap = 48, int iterations = 20, int nun_threads = 8, float coef = 0.01f, bool show_preview = true, bool denoise = false);

};

/** @brief solves all tiles in wavefront (anti-diagonal) order: wave w = i + j,
all tiles of a wave run in parallel and every tile is warm-started from the
already-solved west/north neighbors' overlap strips (falling back to the
generic reference solution when no neighbor exists, e.g. the first wave).
*/
void decrypt_tiles(int num_threads, std::vector<std::vector<cv::Mat>>& mats_in, std::vector<std::vector<indices>> indices,
    std::vector<std::vector<cv::Mat>>& mats_out, const std::vector<std::vector<TileCoord>>& coordinates,
    int num_tiles, int overlap, int iterations, cv::Size tile_size, float coef, bool ycrcb = false);
#endif