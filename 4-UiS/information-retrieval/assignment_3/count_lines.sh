lines=0

for doc in knowledge_base/*.ttl
do
    file_lines=$(wc -l $doc | cut -d " " -f1)
    ((lines+=file_lines))
done

echo $lines
