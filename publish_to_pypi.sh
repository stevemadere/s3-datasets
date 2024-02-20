#!/usr/bin/env bash
#
# if a -t or --test option is provided use a different repository
if [ "$1" == "-t" ] || [ "$1" == "--test" ]; then
  REPOSITORY=testpypi
else
  REPOSITORY=pypi
  exit 1
fi
rm -f dist/s3datasets*
python3 -m build
python3 -m twine upload --repository "$REPOSITORY" dist/s3datasets*
