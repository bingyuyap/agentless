import argparse
import concurrent.futures
import json
import os
from threading import Lock

from tqdm import tqdm

from agentless.fl.Index import EmbeddingIndex
from agentless.util.preprocess_data import (
    filter_none_python,
    filter_out_test_files,
    get_repo_structure,
)
from agentless.util.utils import load_json, load_jsonl, setup_logger


def retrieve_locs(bug, args, local_repo_data, found_files, prev_o, write_lock=None):
    found = False
    for o in prev_o:
        if o["instance_id"] == bug["instance_id"]:
            found = True
            break

    if found:
        logger.info(f"skipping {bug['instance_id']} since patch already generated")
        return None

    instance_id = bug["instance_id"]
    log_file = os.path.join(args.output_folder, "retrieval_logs", f"{instance_id}.log")
    logger = setup_logger(log_file)
    logger.info(f"Processing bug {instance_id}")

    # Get problem statement from local data
    problem_statement = bug["problem_statement"]

    # Get repository structure using local path
    structure = get_repo_structure(
        instance_id,
        args.local_repo_path,  # Use the local path from arguments
        None,  # No commit ID needed for local repos
        "playground",
        is_local=True  # Indicate this is a local repository
    )

    filter_out_test_files(structure)

    if args.filter_file:
        kwargs = {
            "given_files": [x for x in found_files if x["instance_id"] == instance_id][0]["found_files"],
            "filter_top_n": args.filter_top_n,
        }
    else:
        kwargs = {}

    retriever = EmbeddingIndex(
        instance_id,
        structure,
        problem_statement,
        persist_dir=args.persist_dir,
        filter_type=args.filter_type,
        index_type=args.index_type,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        logger=logger,
        **kwargs,
    )

    file_names, meta_infos, traj = retriever.retrieve(mock=args.mock)

    if write_lock is not None:
        write_lock.acquire()
    with open(args.output_file, "a") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "found_files": file_names,
                    "node_info": meta_infos,
                    "traj": traj,
                }
            )
            + "\n"
        )
    if write_lock is not None:
        write_lock.release()

def retrieve(args):
    if args.filter_file:
        found_files = load_jsonl(args.filter_file)
    else:
        found_files = []

    # Load local repository data instead of SWE-bench dataset
    with open(args.local_repo, 'r') as f:
        local_repo_data = json.load(f)

    prev_o = load_jsonl(args.output_file) if os.path.exists(args.output_file) else []

    if args.num_threads == 1:
        for bug in tqdm(local_repo_data, colour="MAGENTA"):  # Use local data
            retrieve_locs(
                bug, args, local_repo_data, found_files, prev_o, write_lock=None
            )
    else:
        write_lock = Lock()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_threads
        ) as executor:
            futures = [
                executor.submit(
                    retrieve_locs,
                    bug,
                    args,
                    local_repo_data,  # Pass local data
                    found_files,
                    prev_o,
                    write_lock,
                )
                for bug in local_repo_data  # Iterate over local data
            ]
            for _ in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(local_repo_data),  # Update total count
                colour="MAGENTA",
            ):
                pass

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="retrieve_locs.jsonl")
    parser.add_argument(
        "--index_type", type=str, default="simple", choices=["simple", "complex"]
    )
    parser.add_argument(
        "--filter_type", type=str, default="none", choices=["none", "given_files"]
    )
    parser.add_argument("--filter_top_n", type=int, default=None)
    parser.add_argument("--filter_file", type=str, default="")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=0)
    parser.add_argument("--persist_dir", type=str)
    parser.add_argument("--target_id", type=str)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for creating API requests (WARNING, embedding token counts are only accurate when thread=1)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        choices=["princeton-nlp/SWE-bench_Lite", "princeton-nlp/SWE-bench_Verified"],
    )
    parser.add_argument(
        "--local_repo",
        type=str,
        required=True,  # Make it required for local operation
        help="Path to local repository JSON file containing bug data"
    )
    parser.add_argument(
        "--local_repo_path",
        type=str,
        required=True,
        help="Path to local repository directory"
    )

    args = parser.parse_args()

    args.output_file = os.path.join(args.output_folder, args.output_file)
    assert (
        not args.filter_type == "given_files" or args.filter_file != ""
    ), "Need to provide a filtering file"

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "retrieval_logs"), exist_ok=True)

    # dump argument
    with open(os.path.join(args.output_folder, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    retrieve(args)


if __name__ == "__main__":
    main()
