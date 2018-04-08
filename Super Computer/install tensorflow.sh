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