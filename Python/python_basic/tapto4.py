import sys

src = sys.argv[1]
dst = sys.argv[2]

with open(src) as f:
    tap_content = f.read()


space_content = tap_content.replace("\t", " "*4)

with open(dst, 'w') as f:
    f.write(space_content)