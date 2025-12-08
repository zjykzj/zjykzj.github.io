# 使用 Node.js 20（LTS）
FROM node:20

# 安装系统依赖：locales、git、时区 + CI/部署所需工具
# --no-install-recommends 减少镜像体积
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        locales \
        git \
        tzdata \
        curl \
        jq \
        openssh-client \
    && \
    # 生成 en_US.UTF-8 locale
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 && \
    # 清理 apt 缓存（减小镜像大小）
    rm -rf /var/lib/apt/lists/*

# 设置 UTF-8 环境变量（关键！避免中文乱码或脚本编码问题）
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# 设置工作目录
WORKDIR /blog

# 全局安装 Hexo CLI
RUN npm install -g hexo-cli

# 默认命令
CMD ["echo", "Use 'hexo init', 'hexo generate', or 'hexo server' as needed."]