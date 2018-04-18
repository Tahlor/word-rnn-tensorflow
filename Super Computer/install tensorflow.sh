mkdir ~/glibc
cd ~/glibc 

wget http://ftp.gnu.org/gnu/glibc/glibc-2.23.tar.gz
tar zxvf glibc-2.23.tar.gz
cd glibc-2.23
mkdir build
cd build

../configure --prefix=/opt/glibc-2.23
make -j4
sudo make install

export LD_LIBRARY_PATH=/opt/glibc-2.23/lib




wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
bash Anaconda3-4.4.0-Linux-x86_64.sh -p $HOME/anaconda3
echo "# Next line makes anaconda my default python, comment out with # to disable this" >> ~/.bashrc
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
rm Anaconda3-4.4.0-Linux-x86_64.sh
source ~/.bashrc
