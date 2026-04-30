import warnings, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # 指定使用第0张显卡
# os.environ["CUDA_VISIBLE_DEVICES"] = '2' # 指定使用第三张显卡
# os.environ["CUDA_VISIBLE_DEVICES"] = '2,3' # 指定使用第三、四张显卡进行多卡训练
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics.models.yolo.detect.afss_train import AFSSDetectionTrainer
from ultralytics.models.yolo.segment.afss_train import AFSSSegmentationTrainer
from ultralytics.models.yolo.pose.afss_train import AFSSPoseTrainer
from ultralytics.models.yolo.obb.afss_train import AFSSOBBTrainer

# BILIBILI UP 魔傀面具
# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

# 全流程实战教程：从零开始完成环境配置、数据集解析、模型训练到测试验证 https://www.bilibili.com/video/BV1tUFkzSEn7/

if __name__ == '__main__':
    yaml_path = '/home/wanglinkai/projects/Ultralytics_305597351/ultralytics/cfg/models/v8/yolov8.yaml'

    # 初始化 YOLO 模型，加载 COCO 预训练权重进行 fine-tune
    model = YOLO('/home/wanglinkai/projects/Ultralytics_305597351/yolov8m.pt')
    # model.load('yolo26n.pt') # 加载预训练权重，一般都不建议加载
    model.train(data='dataset/data.yaml', # 数据集配置文件路径
                cache=False, # 是否缓存图像到内存以加快训练速度。False=不缓存，True=缓存到RAM(很吃内存，内存少的慎开)，'disk'=缓存到磁盘(吃硬盘空间)
                imgsz=640, # 输入图像尺寸（像素）
                epochs=300, # 训练总轮数
                batch=16, # 批次大小
                close_mosaic=0, # 最后多少个 epoch 关闭 Mosaic 数据增强。设置 0 代表全程开启 Mosaic 训练
                workers=0, # 数据加载的工作线程数。Windows 下出现卡顿或奇怪错误可尝试设置为 0
                device='0', # CUDA_VISIBLE_DEVICES已在头部设置，PyTorch视角下只有cuda:0
                optimizer='MuSGD' if 'yolo26' in yaml_path else 'SGD', # 优化器选择。YOLO26 使用官方推荐的 MuSGD，其他模型使用 SGD
                patience=50, # 早停机制的耐心值。连续 50 个 epoch 验证指标未提升则停止训练。设置 0 关闭早停
                # resume=True, # 断点续训，需要在 YOLO 初始化时加载 last.pt 权重文件
                amp=True, # 是否启用自动混合精度（Automatic Mixed Precision）训练，默认为 True | loss出现nan可以关闭amp
                # fraction=0.2, # 设置0.2代表只选择百分之20的数据进行训练
                cos_lr=False, # 是否使用余弦退火学习率调度器，默认为 False
                save_period=-1, # 每隔多少个 epoch 保存一次 checkpoint（默认 -1 表示禁用，仅保存最好和最后的）
                project='train', # 训练结果保存的项目目录
                name='exp', # 本次实验的名称，（若已存在则自动创建 exp2, exp3...）

                # trainer=AFSSDetectionTrainer,
                # afss=True, # 开启 AFSS
                # afss_save_refresh_json=False,
                # afss_warmup_epochs=20, # 前 20 个 epoch 使用全量训练集 warmup
                # afss_update_interval=5, # 每隔 5 个 epoch 刷新一次图像难度状态
                # afss_easy_ratio=0.02, # easy 样本每轮保留 2%
                # afss_moderate_ratio=0.40, # moderate 样本每轮保留 40%
                # afss_easy_forced_gap=10, # easy 样本超过 10 个 epoch 未使用则强制回看
                # afss_moderate_forced_gap=3, # moderate 样本超过 3 个 epoch 未使用则强制覆盖
                # afss_thresholds={
                #     "detect": [0.55, 0.85],
                #     "obb": [0.55, 0.85],
                #     "segment": [0.55, 0.85],
                #     "pose": [0.55, 0.85],
                # },

                # -------------------- LOSS部分(更多解释可以看LOSS-UserGuide.md) --------------------
                cls_loss='bce', # 分类损失类型可选：bce, slide, ema_slide, focal, varifocal, qualityfocal
                iou_loss='ciou', # IoU损失可选：基础 iou/giou/diou/ciou/eiou/siou/shapeiou/piou/piou2；组合 inner_<base>/focaler_<base>；MPD mpdiou/inner_mpdiou/focaler_mpdiou；Wise wiseiou[_inner|_focaler]_<variant>
                iou_aux='none', # IoU辅助分支可选：none, gcd, nwd（none 表示关闭辅助分支）
                iou_aux_ratio=0.5, # IoU主损失与辅助分支混合系数（0~1），仅在 iou_aux != none 时生效 
                )
