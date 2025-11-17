#include "tech_renaissance.h"

using namespace tr;

int main() {
    std::cout << std::fixed << std::setprecision(4);
    Logger::get_instance().set_quiet_mode(true);

    try {
        auto cpu_backend = BackendManager::get_cpu_backend();

#ifdef TR_BUILD_PYTHON_SESSION
        {
            auto script_path = std::string(WORKSPACE_PATH) + "/../python/module/mnist_server.py";
            PythonSession ps(script_path, "mnist", true);
            ps.start();
            std::this_thread::sleep_for(std::chrono::milliseconds(4000));
            std::cout << "GO!!!"  << std::endl  << std::endl;

            // 从Python获取测试数据
            Tensor data = ps.calculate("data", 10000);
            data.summary("data");

            Tensor label = ps.calculate("label", 10000);
            label = cpu_backend->cast(label, DType::INT32);
            label.summary("label");

            Tensor output = ps.calculate("output", 10000);
            output.summary("output");

            Tensor loss = ps.calculate("loss", 10000);
            loss.summary("loss");
            loss.print("loss");

            Tensor samples = ps.calculate("samples", 10000);
            samples.summary("samples");

            // 获取权重
            // PyTorch导出的权重形状是(out_features, in_features)
            // 我们的Linear层期望权重形状是(in_features, out_features)
            // 所以需要转置一次以适配我们的内部格式
            Tensor weights0 = ps.calculate("weights0", 10000);
            cpu_backend->transpose_inplace(weights0);  // 转置为(in_features, out_features)
            weights0.summary("weights0");

            Tensor weights1 = ps.calculate("weights1", 10000);
            cpu_backend->transpose_inplace(weights1);  // 转置为(in_features, out_features)
            weights1.summary("weights1");

            Tensor weights2 = ps.calculate("weights2", 10000);
            cpu_backend->transpose_inplace(weights2);  // 转置为(in_features, out_features)
            weights2.summary("weights2");

            // 验证输入数据一致性
            auto my_samples = cpu_backend->view(data, Shape(4,784));
            my_samples.summary("my_samples");
            if (my_samples.is_view()) {
                std::cout << "my_samples is a view" << std::endl;
            }

            bool ok = cpu_backend->is_close(my_samples, samples);
            if (ok) {
                std::cout << "samples are equal" << std::endl;
            } else {
                std::cout << "samples are NOT equal" << std::endl;
            }

            // ===== 使用Module的MLP实现 =====
            std::cout << "\n=== Testing Module-based MLP ===" << std::endl;

            // 创建MLP模块：Flatten -> Linear(784, 512) -> Tanh -> Linear(512, 256) -> Tanh -> Linear(256, 10)
            Flatten flatten;
            Linear fc1(784, 512, "fc1");
            Tanh act1("tanh1");
            Linear fc2(512, 256, "fc2");
            Tanh act2("tanh2");
            Linear fc3(256, 10, "fc3");

            // 设置后端
            flatten.set_backend(cpu_backend.get());
            fc1.set_backend(cpu_backend.get());
            act1.set_backend(cpu_backend.get());
            fc2.set_backend(cpu_backend.get());
            act2.set_backend(cpu_backend.get());
            fc3.set_backend(cpu_backend.get());

            // 设置权重（从Python获取的权重）
            // 注意：我们的权重形状需要是 (out_features, in_features)
            // Python获取的权重已经是转置后的形状
            // Linear层默认不使用偏置
            fc1.register_parameter("weight", weights0);
            fc2.register_parameter("weight", weights1);
            fc3.register_parameter("weight", weights2);

            // 前向传播
            // 使用Flatten层将输入从(4,1,28,28)展平为(4,784)
            data.summary("original_data");
            Tensor input_reshaped = flatten.forward(data);
            input_reshaped.summary("flatten_output");

            // 使用Module进行前向传播
            Tensor h1 = fc1.forward(input_reshaped);
            h1.summary("fc1 output");

            Tensor h1_activated = act1.forward(h1);
            h1_activated.summary("tanh1 output");

            Tensor h2 = fc2.forward(h1_activated);
            h2.summary("fc2 output");

            Tensor h2_activated = act2.forward(h2);
            h2_activated.summary("tanh2 output");

            Tensor my_results = fc3.forward(h2_activated);
            my_results.summary("my_results_module");

            // 比较结果
            ok = cpu_backend->is_close(my_results, output);
            if (ok) {
                std::cout << "Module outputs are equal to PyTorch outputs" << std::endl;
            } else {
                std::cout << "Module outputs are NOT equal to PyTorch outputs" << std::endl;
            }

            // 计算loss（与原始代码相同的计算方式）
            label.print("label");
            auto oh = cpu_backend->one_hot(label, 10);
            oh.print("one-hot");
            my_results.print("my_results_module");
            auto pred = cpu_backend->softmax(my_results, 1);
            pred.print("pred_module");
            auto my_loss = cpu_backend->crossentropy(pred, oh);
            std::cout << "my_loss_module: " << my_loss << std::endl;
            std::cout << "loss: " << loss.item<float>() << std::endl;

            // 验证loss是否一致
            float loss_diff = std::abs(my_loss - loss.item<float>());
            if (loss_diff < 1e-4f) {
                std::cout << "Module loss matches PyTorch loss (diff: " << loss_diff << ")" << std::endl;
            } else {
                std::cout << "Module loss does NOT match PyTorch loss (diff: " << loss_diff << ")" << std::endl;
            }

            ps.please_exit();
        }
#endif

    } catch (const ShapeError& e) {
        std::cerr << "ShapeError: " << e.what() << std::endl;
        return 1;
    } catch (const TypeError& e) {
        std::cerr << "TypeError: " << e.what() << std::endl;
        return 1;
    } catch (const TRException& e) {
        std::cerr << "TRException: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception occurred" << std::endl;
        return 1;
    }

    return 0;
}