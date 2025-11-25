/**
 * @file test_mnist_prepare.cpp
 * @brief MNIST数据集下载、解压、读取模块测试
 * @details 独立实现的MNIST数据集准备功能，包括下载、解压、读取
 * @author 技术觉醒团队
 */

// UTF-8 BOM to avoid C4828 warning
#pragma execution_character_set("utf-8")

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <cstdint>

// 第三方库
#include <curl/curl.h>
#include <zlib.h>

namespace fs = std::filesystem;

// 简单的Tensor类定义（仅用于测试）
enum class DType {
    FP32,
    INT32
};

struct SimpleTensor {
    std::vector<float> data;  // 简化：只支持float数据
    std::vector<int> shape;   // 形状
    DType dtype;              // 数据类型

    SimpleTensor(const std::vector<int>& shape, DType dtype = DType::FP32)
        : shape(shape), dtype(dtype) {
        size_t total_elements = 1;
        for (int dim : shape) {
            total_elements *= dim;
        }
        if (dtype == DType::FP32) {
            data.resize(total_elements);
        } else {
            // 对于INT32类型，我们存储在float vector中（简化处理）
            data.resize(total_elements);
        }
    }

    size_t size() const {
        return data.size();
    }

    float* ptr() {
        return data.data();
    }

    const float* ptr() const {
        return data.data();
    }
};

// MNIST文件信息结构
struct MnistFileInfo {
    std::string filename;
    std::string primary_url;
    std::string backup_url;
};

// MNIST文件URL配置
const std::vector<MnistFileInfo> MNIST_FILES = {
    {"train-images-idx3-ubyte.gz",
     "https://tech-renaissance.cn/download/mnist/train-images-idx3-ubyte.gz",
     "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"},

    {"train-labels-idx1-ubyte.gz",
     "https://tech-renaissance.cn/download/mnist/train-labels-idx1-ubyte.gz",
     "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz"},

    {"t10k-images-idx3-ubyte.gz",
     "https://tech-renaissance.cn/download/mnist/t10k-images-idx3-ubyte.gz",
     "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz"},

    {"t10k-labels-idx1-ubyte.gz",
     "https://tech-renaissance.cn/download/mnist/t10k-labels-idx1-ubyte.gz",
     "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"}
};

// 工具函数：标准化路径（处理末尾斜杠）
std::string normalize_path(const std::string& path) {
    if (path.empty()) return "./";
    if (path.back() == '/' || path.back() == '\\') return path;
    return path + "/";
}

// 工具函数：大端序转小端序
uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xFF000000) |
           ((val << 8)  & 0x00FF0000) |
           ((val >> 8)  & 0x0000FF00) |
           ((val >> 24) & 0x000000FF);
}

// libcurl写入回调函数
size_t write_callback(void* ptr, size_t size, size_t nmemb, void* stream) {
    std::ofstream* out = static_cast<std::ofstream*>(stream);
    size_t count = size * nmemb;
    out->write(static_cast<char*>(ptr), count);
    return count;
}

// 下载单个文件
bool download_file(const std::string& url, const fs::path& file_path) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "[Error] Failed to initialize libcurl." << std::endl;
        return false;
    }

    std::ofstream out_file(file_path, std::ios::binary);
    if (!out_file.is_open()) {
        std::cerr << "[Error] Cannot open file for writing: " << file_path << std::endl;
        curl_easy_cleanup(curl);
        return false;
    }

    std::cout << "[Info] Downloading from: " << url << " ..." << std::endl;

    // 设置Curl选项
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &out_file);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 15L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);

    // 执行请求
    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    // 清理资源
    curl_easy_cleanup(curl);
    out_file.close();

    if (res == CURLE_OK && http_code == 200) {
        std::cout << "[Success] Download complete: " << file_path.filename() << std::endl;
        return true;
    } else {
        std::cerr << "[Failed] Download failed: " << curl_easy_strerror(res)
                  << " (HTTP " << http_code << ")" << std::endl;
        // 删除不完整的文件
        if (fs::exists(file_path)) {
            fs::remove(file_path);
        }
        return false;
    }
}

// 下载MNIST文件
bool download_mnist(const std::string& dir_name) {
    std::string normalized_dir = normalize_path(dir_name);
    fs::path mnist_dir = fs::path(normalized_dir) / "MNIST" / "raw";

    try {
        // 创建目录
        if (!fs::exists(mnist_dir)) {
            if (!fs::create_directories(mnist_dir)) {
                std::cerr << "[Error] Failed to create directory: " << mnist_dir << std::endl;
                return false;
            }
        }

        std::cout << "[Info] MNIST download directory: " << mnist_dir << std::endl;

        // 下载所有文件
        for (const auto& file_info : MNIST_FILES) {
            fs::path file_path = mnist_dir / file_info.filename;

            // 检查文件是否已存在
            if (fs::exists(file_path)) {
                std::cout << "[Info] File already exists, skipping: " << file_info.filename << std::endl;
                continue;
            }

            // 尝试首选URL
            if (download_file(file_info.primary_url, file_path)) {
                continue; // 下载成功，继续下一个文件
            }

            // 首选URL失败，尝试备用URL
            std::cout << "[Info] Primary URL failed, trying backup URL for: " << file_info.filename << std::endl;
            if (!download_file(file_info.backup_url, file_path)) {
                std::cerr << "[Error] All download attempts failed for: " << file_info.filename << std::endl;
                return false;
            }
        }

        std::cout << "[Success] All MNIST files downloaded successfully!" << std::endl;
        return true;

    } catch (const fs::filesystem_error& e) {
        std::cerr << "[Exception] Filesystem error: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[Exception] Error: " << e.what() << std::endl;
        return false;
    }
}

// 解压单个.gz文件到内存
std::vector<uint8_t> decompress_gz_file(const fs::path& gz_path) {
    gzFile file = gzopen(gz_path.string().c_str(), "rb");
    if (!file) {
        throw std::runtime_error("Cannot open gzip file: " + gz_path.string());
    }

    std::vector<uint8_t> buffer;
    const size_t CHUNK_SIZE = 16384;
    std::vector<uint8_t> temp(CHUNK_SIZE);
    int bytes_read;

    while ((bytes_read = gzread(file, temp.data(), static_cast<unsigned int>(CHUNK_SIZE))) > 0) {
        buffer.insert(buffer.end(), temp.begin(), temp.begin() + bytes_read);
    }

    gzclose(file);

    if (bytes_read < 0) {
        throw std::runtime_error("Error reading gzip file: " + gz_path.string());
    }

    return buffer;
}

// 读取MNIST图像文件
SimpleTensor load_mnist_images(const fs::path& image_path) {
    std::cout << "[Info] Loading images from: " << image_path.filename() << std::endl;

    std::vector<uint8_t> data = decompress_gz_file(image_path);

    if (data.size() < 16) {
        throw std::runtime_error("Invalid MNIST image file: too small");
    }

    // 读取文件头（大端序）
    uint32_t magic = swap_endian(*reinterpret_cast<uint32_t*>(&data[0]));
    uint32_t num_images = swap_endian(*reinterpret_cast<uint32_t*>(&data[4]));
    uint32_t rows = swap_endian(*reinterpret_cast<uint32_t*>(&data[8]));
    uint32_t cols = swap_endian(*reinterpret_cast<uint32_t*>(&data[12]));

    if (magic != 2051) {
        throw std::runtime_error("Invalid magic number for MNIST images: " + std::to_string(magic));
    }

    std::cout << "[Info] Image dataset info: " << num_images << " images, "
              << rows << "x" << cols << " pixels" << std::endl;

    // 创建Tensor并填充数据
    SimpleTensor images({static_cast<int>(num_images), static_cast<int>(rows * cols)}, DType::FP32);

    size_t expected_size = 16 + num_images * rows * cols;
    if (data.size() != expected_size) {
        throw std::runtime_error("MNIST image file size mismatch");
    }

    // 将像素值从uint8_t转换为float并归一化到[0,1]
    for (size_t i = 0; i < num_images * rows * cols; ++i) {
        images.data[i] = static_cast<float>(data[16 + i]) / 255.0f;
    }

    return images;
}

// 读取MNIST标签文件
SimpleTensor load_mnist_labels(const fs::path& label_path) {
    std::cout << "[Info] Loading labels from: " << label_path.filename() << std::endl;

    std::vector<uint8_t> data = decompress_gz_file(label_path);

    if (data.size() < 8) {
        throw std::runtime_error("Invalid MNIST label file: too small");
    }

    // 读取文件头（大端序）
    uint32_t magic = swap_endian(*reinterpret_cast<uint32_t*>(&data[0]));
    uint32_t num_labels = swap_endian(*reinterpret_cast<uint32_t*>(&data[4]));

    if (magic != 2049) {
        throw std::runtime_error("Invalid magic number for MNIST labels: " + std::to_string(magic));
    }

    std::cout << "[Info] Label dataset info: " << num_labels << " labels" << std::endl;

    // 创建Tensor并填充数据
    SimpleTensor labels({static_cast<int>(num_labels)}, DType::INT32);

    size_t expected_size = 8 + num_labels;
    if (data.size() != expected_size) {
        throw std::runtime_error("MNIST label file size mismatch");
    }

    // 将标签值从uint8_t转换为int32
    for (size_t i = 0; i < num_labels; ++i) {
        labels.data[i] = static_cast<float>(data[8 + i]);  // 存储为float但表示int32
    }

    return labels;
}

// 解压并加载MNIST数据集
bool unzip_and_load_mnist(const std::string& dir_name,
                         SimpleTensor& train_images, SimpleTensor& train_labels,
                         SimpleTensor& test_images, SimpleTensor& test_labels) {
    std::string normalized_dir = normalize_path(dir_name);
    fs::path mnist_dir = fs::path(normalized_dir) / "MNIST" / "raw";

    try {
        std::cout << "[Info] Loading MNIST dataset from: " << mnist_dir << std::endl;

        // 加载训练数据
        train_images = load_mnist_images(mnist_dir / "train-images-idx3-ubyte.gz");
        train_labels = load_mnist_labels(mnist_dir / "train-labels-idx1-ubyte.gz");

        // 加载测试数据
        test_images = load_mnist_images(mnist_dir / "t10k-images-idx3-ubyte.gz");
        test_labels = load_mnist_labels(mnist_dir / "t10k-labels-idx1-ubyte.gz");

        std::cout << "[Success] MNIST dataset loaded successfully!" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[Error] Failed to load MNIST dataset: " << e.what() << std::endl;
        return false;
    }
}

// 主函数：准备MNIST数据集
bool prepare_mnist(const std::string& dir_name,
                   SimpleTensor& train_images, SimpleTensor& train_labels,
                   SimpleTensor& test_images, SimpleTensor& test_labels) {

    std::cout << "=== MNIST Dataset Preparation ===" << std::endl;
    std::cout << "[Info] Target directory: " << dir_name << std::endl;

    // 步骤1：下载文件
    std::cout << "\n[Step 1/3] Downloading MNIST files..." << std::endl;
    if (!download_mnist(dir_name)) {
        std::cerr << "[Error] Failed to download MNIST files!" << std::endl;
        return false;
    }

    // 步骤2：解压并读取文件
    std::cout << "\n[Step 2/3] Extracting and loading MNIST dataset..." << std::endl;
    if (!unzip_and_load_mnist(dir_name, train_images, train_labels, test_images, test_labels)) {
        std::cerr << "[Error] Failed to load MNIST dataset!" << std::endl;
        return false;
    }

    // 步骤3：验证数据
    std::cout << "\n[Step 3/3] Verifying dataset integrity..." << std::endl;
    std::cout << "[Info] Train images shape: [" << train_images.shape[0] << ", " << train_images.shape[1] << "]" << std::endl;
    std::cout << "[Info] Train labels shape: [" << train_labels.shape[0] << "]" << std::endl;
    std::cout << "[Info] Test images shape: [" << test_images.shape[0] << ", " << test_images.shape[1] << "]" << std::endl;
    std::cout << "[Info] Test labels shape: [" << test_labels.shape[0] << "]" << std::endl;

    // 显示一些样本数据
    std::cout << "[Info] Sample training labels: ";
    for (int i = 0; i < std::min(10, train_labels.shape[0]); ++i) {
        std::cout << static_cast<int>(train_labels.data[i]) << " ";
    }
    std::cout << std::endl;

    std::cout << "[Success] MNIST dataset preparation completed successfully!" << std::endl;
    return true;
}

// 打印MNIST图像（ASCII可视化）
void print_mnist_image(const SimpleTensor& images, const SimpleTensor& labels, int index, int rows = 28, int cols = 28) {
    if (index < 0 || index >= images.shape[0]) {
        std::cerr << "[Error] Invalid image index: " << index << std::endl;
        return;
    }

    int label = static_cast<int>(labels.data[index]);
    std::cout << "\n--- MNIST Image #" << index << " (Label: " << label << ") ---" << std::endl;

    const float* img_data = images.data.data() + index * rows * cols;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float pixel = img_data[r * cols + c];
            // 根据像素强度选择ASCII字符
            if (pixel > 0.8f) std::cout << "8";
            else if (pixel > 0.6f) std::cout << "#";
            else if (pixel > 0.4f) std::cout << "*";
            else if (pixel > 0.2f) std::cout << ".";
            else std::cout << " ";
        }
        std::cout << std::endl;
    }
}

// 主函数（测试用例）
int main() {
    // 初始化libcurl
    curl_global_init(CURL_GLOBAL_ALL);

    try {
        // 使用WORKSPACE_PATH宏指向workspace目录
        std::string test_dir = WORKSPACE_PATH;

        std::cout << "=== MNIST Preparation Test ===" << std::endl;
        std::cout << "This test will download, extract, and load the MNIST dataset." << std::endl;
        std::cout << "Download directory: " << fs::absolute(test_dir) << std::endl;

        // 准备MNIST数据集
        SimpleTensor train_images({1, 784}), train_labels({1});
        SimpleTensor test_images({1, 784}), test_labels({1});

        if (!prepare_mnist(test_dir, train_images, train_labels, test_images, test_labels)) {
            std::cerr << "[Error] MNIST preparation failed!" << std::endl;
            curl_global_cleanup();
            return 1;
        }

        // 显示前几个样本
        std::cout << "\n=== Sample MNIST Images ===" << std::endl;
        for (int i = 0; i < std::min(5, train_images.shape[0]); ++i) {
            print_mnist_image(train_images, train_labels, i);
            std::cout << std::endl;
        }

        std::cout << "[Success] Test completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[Exception] Test failed with exception: " << e.what() << std::endl;
        curl_global_cleanup();
        return 1;
    }

    // 清理libcurl
    curl_global_cleanup();
    return 0;
}