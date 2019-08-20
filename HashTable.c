#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef char * Key;
typedef char * Value;

#define EMPTY 0
#define SET 1
#define REMOVED 2

const char * Labels[] = {"Empty","Set","Removed"};
const char * Colors[] = {"green!25","red!25","orange!25"};

typedef struct {
   Key key;
   unsigned char status;
   Value val;
} Data;

typedef struct {
  Data * tab;
  int size;
} HashTable;

int Insert(HashTable ht, Key key, Value val);
int Query(HashTable ht, Key key, Value * val, unsigned int *position);
void Init(HashTable *ht, int size);
void Destroy(HashTable ht);
int Delete(HashTable ht, Key key);
void Clean(HashTable ht);
void Stats(HashTable ht);
void ExportPdf(HashTable ht, char * out);
void error(void);

int main(void) 
{
   int size;
   char lecture[100];
   char * key;
   char * val;
   HashTable ht;
   int step=0;

   if (fscanf(stdin,"%99s",lecture)!=1)
      error();
   while (strcmp(lecture,"bye")!=0)
   {
      if (strcmp(lecture,"init")==0)
      {
         if (fscanf(stdin,"%99s",lecture)!=1)
            error();
         size = atoi(lecture);
         /* mettre le code d'initialisation ici */
         Init(&ht,size);
      }
      else if (strcmp(lecture,"insert")==0)
      {
         if (fscanf(stdin,"%99s",lecture)!=1)
            error();
         key = strdup(lecture);
         if (fscanf(stdin,"%99s",lecture)!=1)
            error();
         val = strdup(lecture);
         /* mettre ici le code d'insertion */
         Insert(ht,key,val);
      }
      else if (strcmp(lecture,"delete")==0)
      {
         if (fscanf(stdin,"%99s",lecture)!=1)
            error();
         key = strdup(lecture);
         /* mettre ici le code de suppression */
         Delete(ht,key);
      }
      else if (strcmp(lecture,"query")==0)
      {
         if (fscanf(stdin,"%99s",lecture)!=1)
            error();
         /* mettre ici le code de query */
         unsigned int pos;
         if (Query(ht,lecture,&val,&pos))
         {
            printf("%s\r\n",val);
         }
         else
         {
            printf("Not found\r\n");
         }
      }
      else if (strcmp(lecture,"destroy")==0)
      {
         /* mettre ici le code de destruction */
         Destroy(ht);
      }
      else if (strcmp(lecture,"stats")==0)
      {
         /* mettre ici le code de destruction */
         Stats(ht);
      }
      else if (strcmp(lecture,"clean")==0)
      {
         Clean(ht);
      }

      char name[100];
      /* commenter les 4 lignes ci-dessous pour un comportement normal du programme */
      sprintf(name,"step_%03d.png",step);
      ExportPdf(ht,name);
      system("open tmp.pdf");
      step++;
      if (fscanf(stdin,"%99s",lecture)!=1)
         error();
   }
   return 0;
}

/* placer ici vos définitions de fonctions ou procédures */

void error(void)
{
   printf("input error\r\n");
   exit(0);
}
unsigned int shift_rotate(unsigned int val, unsigned int n)
{
  n = n%(sizeof(unsigned int)*8);
  return (val<<n) | (val>> (sizeof(unsigned int)*8-n));
}

unsigned int Encode(Key key)
{
   unsigned int i;
   unsigned int val = 0;
   unsigned int power = 0;
   for (i=0;i<strlen(key);i++)
   {
     val += shift_rotate(key[i],power*7);
     power++;
   }
   return val;
}

unsigned int hash(unsigned int val, unsigned int size)
{
   return val%size;
}

unsigned int HashFunction(Key key, unsigned int size)
{
   //printf("Hash function of %s : %d\n",key,hash(Encode(key),size));
   return hash(Encode(key),size);
}

int Insert(HashTable ht, Key key, Value val)
{
  int i=0;
  unsigned int hv = HashFunction(key,ht.size);
  Data d;
  Value v;
  unsigned int pos;
  /* if the key is already present, we replace the value */
  if (Query(ht,key,&v,&pos)) {
     ht.tab[pos].val = val;
     return 1;
  }

  do {
     d = ht.tab[(hv+i)%ht.size];
     i++;
  }/* we stop when the cell is EMPTY, REMOVED, SET with the inserted key, or when all cells have been read */
  while (!( d.status == EMPTY || d.status == REMOVED || (d.status==SET && strcmp(key,d.key)==0) || i==ht.size));

  i--;

  /* the table is full ! */
  if (i==ht.size-1)
     return 0;
  
  /* if the status is REMOVED we can safely insert at this place because
   * the Query did not find the key before */
  d.key = key;
  d.status = SET;
  d.val = val;
  ht.tab[(hv+i)%ht.size] = d;
  return 1; 
}

int Query(HashTable ht, Key key, Value * val, unsigned int *position)
{
  int i=0;
  unsigned int hv = HashFunction(key,ht.size);
  Data d;
  do {
     d = ht.tab[(hv+i)%ht.size];
     i++;
  }
  while (!( d.status == EMPTY || (d.status==SET && strcmp(key,d.key)==0) || i==ht.size));
  if (d.status==SET && strcmp(key,d.key)==0)
  {
     *val = d.val;
     *position = (hv+i-1)%ht.size;
     return 1;
  }
  else
     return 0;
}

void Init(HashTable *ht, int size)
{
   ht->tab = malloc(sizeof(Data)*size);
   ht->size = size;
   int i;
   Data d;
   d.status = EMPTY;
   for (i=0;i<size;i++)
      ht->tab[i] = d;
}

void Destroy(HashTable ht)
{
   free(ht.tab);
}

int Delete(HashTable ht, Key key)
{
  int i=0;
  unsigned int hv = HashFunction(key,ht.size);
  Data d;
  do {
     d = ht.tab[(hv+i)%ht.size];
     i++;
  }
  while (!( d.status == EMPTY || (d.status==SET && strcmp(key,d.key)==0) || i==ht.size));
  i--;
  if (d.status==SET && strcmp(key,d.key)==0)
  {
     d.status = REMOVED;
     free(d.key);
     free(d.val);
     ht.tab[(hv+i)%ht.size] = d;
     return 1;
  }
  else
     return 0;
    
}

void Clean(HashTable ht) 
{
   int isColliding[ht.size];
   int i;
   for (i=0;i<ht.size;i++) {
      isColliding[i] = 0;
   }

   /* tag colliding cells */
   Data d;
   for (i=0;i<ht.size;i++) {
      d = ht.tab[i];
      if (d.status==SET)
      {
         unsigned int hv = HashFunction(d.key,ht.size);
         if (hv<i) {
            int j;
            for (j=hv;j<i;j++)
               isColliding[j] = 1;
         }
         else if (hv>i) {
            int j;
            for (j=hv;j<i+ht.size;j++)
               isColliding[j%ht.size] = 1;
         }
      }
   }
   /* set non colliding cells that are REMOVED as EMPTY */
   for (i=0;i<ht.size;i++) {
      d = ht.tab[i];
      if (d.status==REMOVED && !isColliding[i]) {
         d.status = EMPTY;
         ht.tab[i] = d;

      }
   }
}

void Stats(HashTable ht)
{
   int i;
   int stats[3];
   for (i=0;i<3;i++)
      stats[i] = 0;
   printf("size    : %d\r\n",ht.size);
   for (i=0;i<ht.size;i++)
   {
      stats[ht.tab[i].status]++;
   }

   printf("empty   : %d\r\n",stats[0]);
   printf("deleted : %d\r\n",stats[1]);
   printf("used    : %d\r\n",stats[2]);
}


void ExportPdf(HashTable ht, char * out) 
{
   int i;
   FILE * fid;
   fid = fopen("tmp.tex","w");
   fprintf(fid,"\\documentclass[12pt]{article}\n\\usepackage[table]{xcolor}\n\\usepackage[sfdefault]{roboto}\n\\usepackage[T1]{fontenc}\n\\usepackage[utf8]{inputenc}\n\\begin{document}\n\\pagestyle{empty}\n\\Huge\\begin{tabular}{|c|c|c|c|c|}\n\\hline\nIndex&~~State~~&~~Key~~&~h(Key)~&~Value~\\\\\\hline\n");
   for (i=0;i<ht.size;i++)
   {
      if (ht.tab[i].status == SET)
         fprintf(fid,"%d&\\cellcolor{%s}%s&%s&%d&%s\\\\\\hline\n",i,Colors[ht.tab[i].status],Labels[ht.tab[i].status],ht.tab[i].key,HashFunction(ht.tab[i].key,ht.size),ht.tab[i].val);
      else
         fprintf(fid,"%d&\\cellcolor{%s}%s& & & \\\\\\hline\n",i,Colors[ht.tab[i].status],Labels[ht.tab[i].status]);
   }
   fprintf(fid,"\\end{tabular}\n\\end{document}\n");
   fclose(fid);
   system("pdflatex tmp.tex >/dev/null");
   char command[100];
   sprintf(command,"convert -trim +repage -background white -flatten tmp.pdf %s",out);
   system(command);
}

