#include "tech_renaissance.h"

using namespace tr;

int main() {
    auto backend = BackendManager::get_cpu_backend();
    auto mnist = MnistDataset(backend, std::string(WORKSPACE_PATH) + "/../../MNIST/tsr/");
    auto model = Model::create("MLP",
        std::make_shared<Flatten>(),
        std::make_shared<Linear>(784, 512),
        std::make_shared<ReLU>(),
        std::make_shared<Linear>(512, 256),
        std::make_shared<ReLU>(),
        std::make_shared<Linear>(256, 10)
    );
    auto loss_fn = CrossEntropyLoss();
    auto optimizer = SGD(0.1f);
    auto scheduler = ConstantLR(0.1f);
    auto trainer = Trainer(model, loss_fn, optimizer, scheduler);
    auto task = Task(model, mnist, trainer);
    TaskConfig cfg;
    cfg.num_epochs = 20;
    cfg.batch_size = 128;
    task.config(cfg);
    task.run();
    return 0;
}