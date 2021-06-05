#!/bin/bash

DATA_PATH="./data"

while getopts p: flag
do
  case "${flag}" in
    p) DATA_PATH=${OPTARG};;
    *) echo "${flag} is not a recognized flag!"; exit 1;
  esac
done

mkdir -p "$DATA_PATH"
pushd "$DATA_PATH" || exit

if [[ $1 == 'ansim' ]]; then
  # ANSIM DATASET: https://zenodo.org/record/1237703

  if [ ! -d ansim ]; then
    mkdir -p ansim
    pushd ansim || exit

    echo 'DOWNLOADING ANSIM DATASET...'

    if [ ! -f LICENSE ]; then
      wget https://zenodo.org/record/1237703/files/LICENSE
    fi

    if [ ! -f ov1_split1.zip ]; then
      wget https://zenodo.org/record/1237703/files/ov1_split1.zip
    fi
    unzip ov1_split1.zip && rm ov1_split1.zip

    if [ ! -f ov1_split2.zip ]; then
      wget https://zenodo.org/record/1237703/files/ov1_split2.zip
    fi
    unzip ov1_split2.zip && rm ov1_split2.zip

    if [ ! -f ov1_split3.zip ]; then
      wget https://zenodo.org/record/1237703/files/ov1_split3.zip
    fi
    unzip ov1_split3.zip && rm ov1_split3.zip

    if [ ! -f ov2_split1.zip ]; then
      wget https://zenodo.org/record/1237703/files/ov2_split1.zip
    fi
    unzip ov2_split1.zip && rm ov2_split1.zip

    if [ ! -f ov2_split2.zip ]; then
      wget https://zenodo.org/record/1237703/files/ov2_split2.zip
    fi
    unzip ov2_split2.zip && rm ov2_split2.zip

    if [ ! -f ov2_split3.zip ]; then
      wget https://zenodo.org/record/1237703/files/ov2_split3.zip
    fi
    unzip ov2_split3.zip && rm ov2_split3.zip

    if [ ! -f ov3_split1.zip ]; then
      wget https://zenodo.org/record/1237703/files/ov3_split1.zip
    fi
    unzip ov3_split1.zip && rm ov3_split1.zip

    if [ ! -f ov3_split2.zip ]; then
      wget https://zenodo.org/record/1237703/files/ov3_split2.zip
    fi
    unzip ov3_split2.zip && rm ov3_split2.zip

    if [ ! -f ov3_split3.zip ]; then
      wget https://zenodo.org/record/1237703/files/ov3_split3.zip
    fi
    unzip ov3_split3.zip && rm ov3_split3.zip

    popd || exit
  fi
elif [[ $1 == 'resim' ]]; then
  # RESIM DATASET: https://zenodo.org/record/1237707

  if [ ! -d resim ]; then
    mkdir -p resim
    pushd resim || exit

    echo 'DOWNLOADING RESIM DATASET...'

    if [ ! -f LICENSE ]; then
      wget https://zenodo.org/record/1237707/files/LICENSE
    fi

    if [ ! -f ov1_split1.zip ]; then
      wget https://zenodo.org/record/1237707/files/ov1_split1.zip
    fi
    unzip ov1_split1.zip && rm ov1_split1.zip

    if [ ! -f ov1_split2.zip ]; then
      wget https://zenodo.org/record/1237707/files/ov1_split2.zip
    fi
    unzip ov1_split2.zip && rm ov1_split2.zip

    if [ ! -f ov1_split3.zip ]; then
      wget https://zenodo.org/record/1237707/files/ov1_split3.zip
    fi
    unzip ov1_split3.zip && rm ov1_split3.zip

    if [ ! -f ov2_split1.zip ]; then
      wget https://zenodo.org/record/1237707/files/ov2_split1.zip
    fi
    unzip ov2_split1.zip && rm ov2_split1.zip

    if [ ! -f ov2_split2.zip ]; then
      wget https://zenodo.org/record/1237707/files/ov2_split2.zip
    fi
    unzip ov2_split2.zip && rm ov2_split2.zip

    if [ ! -f ov2_split3.zip ]; then
      wget https://zenodo.org/record/1237707/files/ov2_split3.zip
    fi
    unzip ov2_split3.zip && rm ov2_split3.zip

    if [ ! -f ov3_split1.zip ]; then
      wget https://zenodo.org/record/1237707/files/ov3_split1.zip
    fi
    unzip ov3_split1.zip && rm ov3_split1.zip

    if [ ! -f ov3_split2.zip ]; then
      wget https://zenodo.org/record/1237707/files/ov3_split2.zip
    fi
    unzip ov3_split2.zip && rm ov3_split2.zip

    if [ ! -f ov3_split3.zip ]; then
      wget https://zenodo.org/record/1237707/files/ov3_split3.zip
    fi
    unzip ov3_split3.zip && rm ov3_split3.zip

    popd || exit
  fi
elif [[ $1 == 'real' ]]; then
  # REAL DATASET: https://zenodo.org/record/1237793

  if [ ! -d real ]; then
    mkdir -p real
    pushd real || exit

    echo 'DOWNLOADING REAL DATASET...'

    if [ ! -f LICENSE ]; then
      wget https://zenodo.org/record/1237793/files/LICENSE
    fi

    if [ ! -f ov1_split1.zip ]; then
      wget https://zenodo.org/record/1237793/files/ov1_split1.zip
    fi
    unzip ov1_split1.zip && rm ov1_split1.zip

    if [ ! -f ov1_split8.zip ]; then
      wget https://zenodo.org/record/1237793/files/ov1_split8.zip
    fi
    unzip ov1_split8.zip && rm ov1_split8.zip
    mv desc_ov1_split8 desc_ov1_split2
    mv wav_ov1_split8_30db wav_ov1_split2_30db

    if [ ! -f ov1_split9.zip ]; then
      wget https://zenodo.org/record/1237793/files/ov1_split9.zip
    fi
    unzip ov1_split9.zip && rm ov1_split9.zip
      mv desc_ov1_split9 desc_ov1_split3
    mv wav_ov1_split9_30db wav_ov1_split3_30db

    if [ ! -f ov2_split1.zip ]; then
      wget https://zenodo.org/record/1237793/files/ov2_split1.zip
    fi
    unzip ov2_split1.zip && rm ov2_split1.zip

    if [ ! -f ov2_split8.zip ]; then
      wget https://zenodo.org/record/1237793/files/ov2_split8.zip
    fi
    unzip ov2_split8.zip && rm ov2_split8.zip
    mv desc_ov2_split8 desc_ov2_split2
    mv wav_ov2_split8_30db wav_ov2_split2_30db

    if [ ! -f ov2_split9.zip ]; then
      wget https://zenodo.org/record/1237793/files/ov2_split9.zip
    fi
    unzip ov2_split9.zip && rm ov2_split9.zip
    mv desc_ov2_split9 desc_ov2_split3
    mv wav_ov2_split9_30db wav_ov2_split3_30db

    if [ ! -f ov3_split1.zip ]; then
      wget https://zenodo.org/record/1237793/files/ov3_split1.zip
    fi
    unzip ov3_split1.zip && rm ov3_split1.zip

    if [ ! -f ov3_split8.zip ]; then
      wget https://zenodo.org/record/1237793/files/ov3_split8.zip
    fi
    unzip ov3_split8.zip && rm ov3_split8.zip
    mv desc_ov3_split8 desc_ov3_split2
    mv wav_ov3_split8_30db wav_ov3_split2_30db

    if [ ! -f ov3_split9.zip ]; then
      wget https://zenodo.org/record/1237793/files/ov3_split9.zip
    fi
    unzip ov3_split9.zip && rm ov3_split9.zip
    mv desc_ov3_split9 desc_ov3_split3
    mv wav_ov3_split9_30db wav_ov3_split3_30db

    popd || exit
  fi
else
  echo 'Please provide the name of the dataset you want to download (ansim, resim or real).'
fi
