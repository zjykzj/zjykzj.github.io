# 使用 Node.js 20（LTS）
FROM node:20

# 安装 git 和时区数据（tzdata 用于 /etc/localtime）
RUN apt-get update && \
    apt-get install -y --no-install-recommends git tzdata && \
    rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /blog

# 全局安装 Hexo CLI（以 root 身份安装，但后续切换用户）
RUN npm install -g hexo-cli

# 默认命令（可选）
CMD ["echo", "Use 'hexo init', 'hexo generate', or 'hexo server' as needed."]