# 使用 Node.js 20（LTS）
FROM node:20

# 安装 locales、git 和时区数据
# --no-install-recommends 减少体积，locales 是关键！
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        locales \
        git \
        tzdata && \
    # 生成 en_US.UTF-8 locale（通用且兼容性好）
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 && \
    # 清理缓存
    rm -rf /var/lib/apt/lists/*

# 设置 UTF-8 环境变量（关键！）
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# 设置工作目录
WORKDIR /blog

# 全局安装 Hexo CLI
RUN npm install -g hexo-cli

# 默认命令（可选）
CMD ["echo", "Use 'hexo init', 'hexo generate', or 'hexo server' as needed."]