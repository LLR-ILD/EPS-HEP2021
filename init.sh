#!/bin/bash
CODE_DIR=$(dirname $(realpath $0))/code

activate_environment () {
    VENV=$CODE_DIR/.venv
    venv_activate=$VENV/bin/activate
    if ! [ -f $venv_activate ]; then
        echo "No python venv found: at " $venv_activate
        echo "It will be built now."
        python3 -m venv $VENV --prompt $(basename $(pwd))
        source $VENV/bin/activate
        pip install --upgrade pip wheel
        pip install -r $CODE_DIR/requirements.txt
        pre-commit install
    else
        source $venv_activate
    fi
}

activate_environment

# -----------------------------------------------------------------------------
#
# Get the large root per-event files.
#
if [ "$1" = "" ]; then
    :
elif [ "$1" = "rootfiles" ]; then
    mkdir -p $CODE_DIR/tmp
    rootfile_destination=$CODE_DIR/tmp/per_event_data
    if [ -d $rootfile_destination ]; then
        echo "$rootfile_destination already exists. If it is empty or with wrong data, "
        echo "please remove the folder and run this script again."
    elif [ "$2" = "local" ]; then
        local_data="$3"
        if [ "$local_data" = "" ]; then
            local_data=$DATA/EPS-HEP2021
        fi
        if ! [ -e $local_data ]; then
            echo "The data source does not exist: " $local_data
            echo "Try again with `./init.sh rootfiles link SOURCE_FOLDER`"
            exit 1
        fi
        ln -s $DATA/EPS-HEP2021 $rootfile_destination
    elif [ "$2" = "remote" ]; then
        server_path=/data_ilc/flc/kunath/LLR-ILD/EPS-HEP2021/
        if ssh  -o ConnectTimeout=1 llrgate01.in2p3.fr "[ -d $server_path ]"; then
            echo "Start copying contents from $CODE_DIR"
            scp -r llrgate01.in2p3.fr:$server_path $rootfile_destination
        else
            echo "If the connection times out, you might not have ssh access to llrgate01.in2p3.fr?"
            echo "Else, please check manually that $server_path exists on the remote."
        fi
    else
        echo "Unexpected second argument: $2."
    fi
    for tar_path in $rootfile_destination/*.tar.gz; do
        tar_name=$(basename $tar_path .tar.gz)
        if ! [ -d $rootfile_destination/$tar_name ]; then
            tar -xzf $tar_path --directory $rootfile_destination/
        fi
        if [ -d $rootfile_destination/$tar_name ]; then
            rm $tar_path
        fi
    done
else
    echo "Unexpected first argument: $1."
fi
