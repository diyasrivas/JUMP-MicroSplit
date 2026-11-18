#!/bin/bash
# cpg0029 ChromaLive + Hoechst (BR00122249): see ../cpg0029_shared/launchers/predict.sh --help
exec "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/../cpg0029_shared/launchers/predict.sh" --dye_set chromalive "$@"
