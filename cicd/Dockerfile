FROM jenkins/jenkins:lts-jdk17

ENV JENKINS_UC_DOWNLOAD="https://ftp-chi.osuosl.org/pub/jenkins"
RUN jenkins-plugin-cli -p workflow-aggregator git pipeline-graph-analysis pipeline-stage-view

USER root

# Add Docker's official GPG key:
RUN install -m 0755 -d /etc/apt/keyrings
RUN curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
RUN chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
RUN echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install docker cli
RUN apt-get update && apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin

# Install python
RUN apt install -y python3 python3-pip python3-venv

