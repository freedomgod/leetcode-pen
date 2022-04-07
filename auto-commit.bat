@echo off
echo 当前路径 E:\我的资料\文章备份\leetcode-pen
cd E:\我的资料\文章备份\leetcode-pen

echo 状态
git status

echo 添加所有修改过的文件
git add .

echo 提交所有修改
git commit -m "update"

echo 推送远程
git push

pause