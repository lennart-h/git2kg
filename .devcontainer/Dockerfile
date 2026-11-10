FROM astral/uv:0.10.2-python3.13-trixie-slim
WORKDIR /workspaces/git2kg

# Install git (and gpg for signed commits: https://code.visualstudio.com/remote/advancedcontainers/sharing-git-credentials#_sharing-gpg-keys)
RUN apt-get update && apt-get -y install git gnupg2


# Install dependencies for manim
RUN apt-get update && apt-get -y install build-essential python3-dev libcairo2-dev libpango1.0-dev ffmpeg