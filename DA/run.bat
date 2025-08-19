@echo off
echo 开始运行所有脚本...

conda activate czsl
:: 依次运行每个 Python 脚本
python train.py --source source1kn --target target1kn
python train.py --source source1unk --target target1unk
python train.py --source source2kn --target target2kn
python train.py --source source2unk --target target2unk
python train.py --source source3kn --target target3kn
python train.py --source source3unk --target target3unk
python train.py --source source4kn --target target4kn
python train.py --source source4unk --target target4unk

echo 全部脚本执行完毕。
pause
