#include "tech_renaissance.h"

using namespace tr;

int main() {
    std::cout << std::fixed << std::setprecision(4);
    Logger::get_instance().set_quiet_mode(true);

    auto cpu_backend = BackendManager::get_cpu_backend();

#ifdef TR_BUILD_PYTHON_SESSION
    {
        auto script_path = std::string(WORKSPACE_PATH) + "/../python/module/mnist_server.py";
        PythonSession ps(script_path, "mnist", false);
        ps.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(4000));
        std::cout << "GO!!!"  << std::endl  << std::endl;

        Tensor data = ps.calculate("data", 10000);
        data.summary("data");
        // data.print("data");

        Tensor label = ps.calculate("label", 10000);
        label.summary("label");
        label.print("label");

        Tensor output = ps.calculate("output", 10000);
        output.summary("output");
        output.print("output");

        Tensor loss = ps.calculate("loss", 10000);
        loss.summary("loss");
        loss.print("loss");

        Tensor samples = ps.calculate("samples", 10000);
        samples.summary("samples");

        auto my_samples = cpu_backend->reshape(data, Shape(2,784));
        my_samples.summary("my_samples");

        bool ok = cpu_backend->is_close(my_samples, samples);
        std::cout << "ok: " << ok << std::endl;
        // Tensor data = ps.fetch_tensor("data", 10000);
        // Tensor label = ps.fetch_tensor("label");
        // Tensor output = ps.fetch_tensor("output");
        // Tensor loss = ps.fetch_tensor("loss");
        //
        // label.summary("label");
        // label.print("label");
        // output.summary("output");
        // output.print("output");
        // loss.summary("loss");
        // loss.print("loss");

        ps.please_exit();

    }
#endif

    return 0;
}