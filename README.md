# docker-traefik-ubuntu-setup
What I did to setup my Ubuntu Server for Docker and Traefik
```
sudo apt-get update
sudo apt-get upgrade
sudo snap install microk8s --classic
sudo apt-get install nano curl wget build-essential
echo "Installing git"
sudo apt-get install git
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
cat ~/.ssh/id_rsa.pub
git config --global user.name "Your Name"
git config --global user.email your_email@example.com
echo "Installing docker"
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
docker --version
sudo usermod -aG docker $USER
sudo curl -L "https://github.com/docker/compose/releases/download/1.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
echo "Cloning gits"
git clone git@github.com:bethington/docker-traefik-ubuntu-setup.git docker
