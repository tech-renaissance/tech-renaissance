#include "tech_renaissance.h"

using namespace tr;

int main() {
    Model lenet{
        Conv(1, 6, 5, 1, 0),  // in_channels = 1, out_channels = 6, kernel_size = 5, stride = 1, padding = 0
        Tanh(),
        AvgPool(),  // kernel_size=2, stride = 2 by default
        Conv(6, 16, 5, 1, 0),
        Tanh(),
        AvgPool(),
        Flatten(),
        Linear(400, 120),
        Tanh(),
        Linear(120, 84),
        Tanh(),
        Linear(84, 10),
        Softmax()
    };

    Dataset data = DataLoader::getinstance()->load(Dataset::MNIST, "./MNIST");
    // 设置训练参数
    Trainer trainer({
        .batch_size = 128,
        .learning_rate = 0.01,
        .momentum = 0.9,
        .weight_decay = 0.0005,
        .lr_scheduler = LRScheduler::StepLR(0.1, 10),
        .device = Device::CPU
    });
    // 模块要提供initialize方法

    // 报告精度、Loss
    // 每XXbatch报告一次训练精度、Loss
    // 每个epoch报告一次测试精度、Loss
    // 估算剩余时间
    // 显示进度条
    // 导出全部原始数据
    // 保存最佳模型
    // 报告当前学习率
    // 报告最佳测试准确率

    // QAT
    // PTQ
    // 浮点训练
    // 浮点测试
    // 量化推理
    dataaugment
    lenet.set_trainer
    Task train_and_val(
        TrainAndVal(20),
        );

    Task task({
        .save_best_model = true,
        .num_epochs = 20,
        .train_report = true,
        .qat = false,
        .device = Device::CPU,
        .model = lenet,
        .data = data,
        .trainer = trainer,
        .val_data = data,
        .val_batch_size = 128,
        .val_report = true,
        .val_report_interval = 100,
        .val_report_time_interval = 1,
        .val_report_progress_bar = true,
        .val_report_save_all_data = true,
        .val_report_save_best_model = true,
    });

    // task.run(20);  // 运行前20个epoch
    task.start();
    // task是一个调度器，它调用trainer的方法，它自己本身不执行算法
    // QAT是trainer的功能
    // task也必须是线程安全的单例模式
    task.enable_saving_best_model();
    for (task.start(); task.ongoing();) {
        task.train_one_epoch();
        task.val();
    }
    task.final_report();

    Ensemble esb(&model1, &model2);
    esb.add_model_pointer(&model3);
    // 数量为偶数会警告，但是能用，因为是取随机数
    // 一次是<的最大值，一次是<=的最大值
    // 然后随机选取
    // random_merge()函数，随机混合两个张量
    task.set_model(esb);
    task.val();

    task.ensemble();
    return 0;
}
