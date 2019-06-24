# docker-traefik-ubuntu-setup
What I did to setup my Ubuntu Server for Docker and Traefik
## Updating Ubuntu
```
sudo apt-get update
sudo apt-get upgrade
sudo snap install microk8s --classic
sudo apt-get install nano curl wget build-essential
```
## Installing git
```
sudo apt-get install git
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
cat ~/.ssh/id_rsa.pub
git config --global user.name "Your Name"
git config --global user.email your_email@example.com
```
## Installing docker
```
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
docker --version
sudo usermod -aG docker $USER
sudo curl -L "https://github.com/docker/compose/releases/download/1.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
exit
```
## Adding environment variable
```
sudo su -
id
echo 'PUID=1000' >> /etc/environment
echo 'PGID=115' >> /etc/environment
echo 'TZ="America/Denver"' >> /etc/environment
echo 'USERDIR="/home/bethington"' >> /etc/environment
echo 'HTTP_USERNAME=username' >> /etc/environment
echo 'HTTP_PASSWORD=mystrongpassword' >> /etc/environment
echo 'DOMAINNAME=example.com' >> /etc/environment
echo 'GODADDY_API_KEY=GODADDY_API_KEY' >> /etc/environment
echo 'GODADDY_API_SECRET=GODADDY_API_SECRET' >> /etc/environment
exit
exit
```
## Cloning gits
```
git clone git@github.com:bethington/docker-traefik-ubuntu-setup.git docker
```
