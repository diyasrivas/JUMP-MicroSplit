#!/bin/bash
# cpg0029 ChromaLive + Hoechst (BR00122249): see ../cpg0029_shared/launchers/train.sh --help
exec "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/../cpg0029_shared/launchers/train.sh" --dye_set chromalive "$@"
