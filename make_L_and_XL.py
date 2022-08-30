import os
import glob

import librosa
import soundfile as sf
import numpy as np
import argparse


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make musdb-L and musdb-XL dataset from its ratio data"
    )

    parser.add_argument(
        "--save_dir", type=str, default="", help="path to save musdb-L and XL dataset"
    )
    parser.add_argument(
        "--musdb_hq_root",
        type=str,
        default="",
        help="path where musdb-hq dataset exists. Make sure that the data is not standard musdb18 but musdb-hq",
    )
    parser.add_argument(
        "--L_XL_ratio_root",
        type=str,
        default="",
        help='path where "musdb-hq vs musdb-L or XL ratio" exists. You should download this on Zenodo first.',
    )
    parser.add_argument(
        "--only_XL",
        type=str2bool,
        default=False,
        help='if you are interested in making only musdb-XL dataset, make this "True"',
    )

    args, _ = parser.parse_known_args()

    musdb_hq_list = glob.glob(
        f"{args.musdb_hq_root}/test/*/mixture.wav"
    )  # we will only make test dataset!

    for path in musdb_hq_list:
        song_name = path.split("/")[-2]
        print(f"processing {song_name}")
        if song_name == "PR - Oh No":
            """
            In case of 'PR - Oh No' track, we used a linear mixture as a mixture, not the original "mixture.wav" file.
            Because the original "mixture.wav" sources are left panned, the limiter is triggered by left channel's energy dominantly.
            This is a very rare case in commercial popular music, so we used a linear mixture as a mixture, which is more natural.
            Please refer https://sigsep.github.io/datasets/musdb.html#errata
            """
            musdb_hq_song = (
                librosa.load(
                    path.replace("mixture.wav", "bass.wav"), sr=None, mono=False
                )[0]
                + librosa.load(
                    path.replace("mixture.wav", "drums.wav"), sr=None, mono=False
                )[0]
                + librosa.load(
                    path.replace("mixture.wav", "other.wav"), sr=None, mono=False
                )[0]
                + librosa.load(
                    path.replace("mixture.wav", "vocals.wav"), sr=None, mono=False
                )[0]
            )
            sr = 44100
        else:
            musdb_hq_song, sr = librosa.load(path, sr=None, mono=False)

        musdb_XL_vs_hq_ratio = np.load(
            f"{args.L_XL_ratio_root}/musdb_XL_ratio/{song_name}.npy"
        )
        musdb_XL_song = musdb_XL_vs_hq_ratio * musdb_hq_song
        os.makedirs(f"{args.save_dir}/musdb_XL/{song_name}", exist_ok=True)
        sf.write(
            f"{args.save_dir}/musdb_XL/{song_name}/mixture.wav",
            musdb_XL_song.T,
            sr,
            subtype="PCM_16",
        )

        if not args.only_XL:
            musdb_L_vs_hq_ratio = np.load(
                f"{args.L_XL_ratio_root}/musdb_L_ratio/{song_name}.npy"
            )
            musdb_L_song = musdb_L_vs_hq_ratio * musdb_hq_song
            os.makedirs(f"{args.save_dir}/musdb_L/{song_name}", exist_ok=True)
            sf.write(
                f"{args.save_dir}/musdb_L/{song_name}/mixture.wav",
                musdb_L_song.T,
                sr,
                subtype="PCM_16",
            )

        for target in ["bass.wav", "drums.wav", "other.wav", "vocals.wav"]:
            target_path = path.replace("mixture.wav", target)
            musdb_hq_target, sr = librosa.load(target_path, sr=None, mono=False)
            musdb_XL_target = musdb_XL_vs_hq_ratio * musdb_hq_target
            sf.write(
                f"{args.save_dir}/musdb_XL/{song_name}/{target}",
                musdb_XL_target.T,
                sr,
                subtype="PCM_16",
            )

            if not args.only_XL:
                musdb_L_target = musdb_L_vs_hq_ratio * musdb_hq_target
                sf.write(
                    f"{args.save_dir}/musdb_L/{song_name}/{target}",
                    musdb_L_target.T,
                    sr,
                    subtype="PCM_16",
                )
