for f in graphs/*; do 
	name=$(basename $f);
	neato -Tpng $f -o img/$name.png; 
done
	