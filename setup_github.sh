#!/bin/bash
# GitHub 仓库上传脚本

echo "=========================================="
echo "FatPIN GitHub 仓库设置脚本"
echo "=========================================="
echo ""

# 检查 git 配置
if [ -z "$(git config user.name)" ] || [ -z "$(git config user.email)" ]; then
    echo "需要设置 Git 用户信息："
    echo ""
    read -p "请输入您的用户名 (GitHub用户名): " username
    read -p "请输入您的邮箱 (GitHub邮箱): " email
    
    git config user.name "$username"
    git config user.email "$email"
    echo ""
    echo "✓ Git 用户信息已设置"
    echo ""
fi

# 提交代码
echo "正在提交代码..."
git add .
git commit -m "Initial commit: FatPIN reinforcement learning code with PortPy support"

if [ $? -eq 0 ]; then
    echo "✓ 代码提交成功"
    echo ""
    
    echo "=========================================="
    echo "下一步操作："
    echo "=========================================="
    echo ""
    echo "1. 在 GitHub 上创建一个新仓库:"
    echo "   - 访问 https://github.com/new"
    echo "   - 输入仓库名称 (例如: fatpin)"
    echo "   - 选择 Public 或 Private"
    echo "   - 不要初始化 README (已经有 README 了)"
    echo ""
    echo "2. 推送代码到 GitHub:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
    echo "   git push -u origin main"
    echo ""
    echo "   或者使用 SSH:"
    echo "   git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git"
    echo "   git push -u origin main"
    echo ""
else
    echo "✗ 代码提交失败"
    exit 1
fi

