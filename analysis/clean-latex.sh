#!/bin/bash
exts="aux bbl blg brf idx ilg ind lof log lol lot out toc synctex.gz"

for fname in *.tex; do
	fname=`basename $fname .tex`
	for ext in $exts; do
		rm -f $fname.$ext
	done
done

