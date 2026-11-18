#!/bin/bash
# cpg0029 Standard CP exposed to ChromaLive (BR00122248): see ../cpg0029_shared/launchers/datasets.sh --help
exec "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/../cpg0029_shared/launchers/datasets.sh" --dye_set post_chromalive_cp "$@"
