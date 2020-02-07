#! /bin/bash
# A sript to download data for the PET part of the SIRF-Exercises course
#
# Usage:
#   /path/download_PET_data.sh optional_destination_directory
# if no argument is used, the destination directory will be set to ~/data
#
# Author: Kris Thielemans, Richard Brown
# Copyright (C) 2018 University College London

# a function to download a file and check its md5
# assumes that file.md5 exists and that the URL env variable is set
function download {
    URL=$1
    filename=$2
    suffix=$3
    if [ -r ${filename} ]
    then
        if md5sum -c ${filename}.md5
        then
            echo "File exists and its md5sum is ok"
            return 0
        else
            echo "md5sum doesn't match. Redownloading."
            rm ${filename}
        fi
    fi

    curl -L -o ${filename} ${URL}${filename}${suffix}

    if md5sum -c ${filename}.md5
    then
        echo "Downloaded file's md5sum is ok"
    else
        echo "md5sum doesn't match. Re-execute this script for another attempt."
        exit 1
    fi
}

set -e
trap "echo some error occured. Retry" ERR

destination=${1:-~/data}

mkdir -p ${destination}
cd ${destination}

# Get file 1 from dropbox
URL=https://www.dropbox.com/s/e5nwpne912n86am/
filename1=sinogram_centered_channels100_140.h5
suffix=?dl=0
rm -f ${filename1}.md5 # (re)create md5 checksum
echo "fccb00c33969c3a3ee6dbb915cdc923e ${filename1}" > ${filename1}.md5
download ${URL} ${filename1} ${suffix}

URL=https://www.ccpi.ac.uk/sites/www.ccpi.ac.uk/files/
filename2=dark.nxs
suffix=?dl=0
rm -f ${filename2}.md5 # (re)create md5 checksum
echo "d5bab6a508e7c5f36dd9803d89b9655b ${filename2}" > ${filename2}.md5
download ${URL} ${filename2}

URL=https://www.ccpi.ac.uk/sites/www.ccpi.ac.uk/files/
filename3=flat.nxs
rm -f ${filename3}.md5 # (re)create md5 checksum
echo "cf6d244b15140479495801031115ed90 ${filename3}" > ${filename3}.md5
download ${URL} ${filename3}

filename4=proj.nxs
rm -f ${filename4}.md5 # (re)create md5 checksum
echo "9f8688e8e77c7677ef47f39b655a6ae8 ${filename4}" > ${filename4}.md5
download ${URL} ${filename4} ${suffix}

#URL=https://www.dropbox.com/s/g04vy3k4ao92zs2/
#filename5=sino_ideal.nxs
#suffix=?dl=0
#rm -f ${filename5}.md5 # (re)create md5 checksum
#echo "75a60f202b6c06457634cc45f598271e ${filename5}" > ${filename5}.md5
#download ${URL} ${filename5} ${suffix}

URL=https://www.ccpi.ac.uk/sites/www.ccpi.ac.uk/files/
filename5=sino_ideal.nxs
rm -f ${filename5}.md5 # (re)create md5 checksum
echo "3c5851d81a67123d3a0a508d3e05158a ${filename5}" > ${filename5}.md5
download ${URL} ${filename5}

# Get Zenodo dataset
# URL=https://zenodo.org/record/2633785/files/
# filenameGRAPPA=PTB_ACRPhantom_GRAPPA.zip
# # (re)download md5 checksum
# echo "a7e0b72a964b1e84d37f9609acd77ef2 ${filenameGRAPPA}" > ${filenameGRAPPA}.md5
# download ${URL} ${filenameGRAPPA}

# echo "Unpacking $filenameGRAPPA"
# unzip -o ${filenameGRAPPA}

# make symbolic links in the normal demo directory
if test -z "$SIRF_INSTALL_PATH"
then
    echo "SIRF_INSTALL_PATH environment variable not set. Using sys.prefix."
    SIRF_INSTALL_PATH=`python -c "from __future__ import print_function; import sys; print (sys.prefix)"`
fi

final_dest=$SIRF_INSTALL_PATH/share/ccpi
echo "Creating symbolic links in ${final_dest} "
mkdir -p ${final_dest}
cd ${final_dest}
rm -f ${filename1}
ln -s ${destination}/${filename1}
rm -f ${filename2}
ln -s ${destination}/${filename2}
rm -f ${filename3}
ln -s ${destination}/${filename3}
rm -f ${filename4}
ln -s ${destination}/${filename4}
rm -f ${filename5}
ln -s ${destination}/${filename5}
echo "All done!"
