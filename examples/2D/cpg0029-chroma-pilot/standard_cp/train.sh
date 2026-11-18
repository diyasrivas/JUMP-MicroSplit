#!/bin/bash
# cpg0029 Standard CP (BR00122250): see ../cpg0029_shared/launchers/train.sh --help
exec "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/../cpg0029_shared/launchers/train.sh" --dye_set standard_cp "$@"
