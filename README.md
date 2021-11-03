# Multi-thread Maximum Clique



## Indice



1.	Introduzione
2.	Algoritmo Sequenziale
3.	Algoritmo Parallelo
a.	Primo Approccio
b.	Secondo Approccio
i.	Senza librerie
ii.	Con libreria Thrust

4.	Dati









## Introduzione

In questo progetto si è parallelizzato l’algoritmo per la ricerca della cricca massima all’interno di un grafo generato randomicamente, quest’ultimo rappresentato con liste di adiacenza.
Lo sviluppo del progetto si è svolto su una macchina che monta una scheda grafica Nvidia modello 2060RTX.
La versione di CUDA utilizzata è 10.2.


## Algoritmo Sequenziale

L’algoritmo sequenziale è un algoritmo ricorsivo e consiste in una ricerca in un grafo che si svolge come segue:
1.	Si sceglie un nodo di partenza, lo si mette in una lista Clique Candidata e si aggiungono tutti i suoi vicini ad una lista Intersezione
2.	In loop si estrae un nodo dalla lista Intersezione, lo si inserisce in Clique Candidata e si chiama la funzione ricorsiva sul nodo estratto
3.	All’inizio della funzione ricorsiva si crea una nuova lista Intersezione facendo appunto l’intersezione tra la lista Intersezione al passo precedente e la lista dei nodi adiacenti al nodo attualmente visitato
4.	Se Intersezione non è vuoto si ritorna al punto 2 altrimenti:
5.	Se la lunghezza di Clique Candidata è maggiore della lunghezza di Clique Massima, allora Clique Massima = Clique Candidata.

Questo algoritmo viene eseguito facendo partire la ricerca da ogni nodo del grafo.
Clique Massima è settato inizialmente a lista vuota, quindi ha lunghezza iniziale a 0.
È stato aggiunto un cut-off al punto 4: se Intersezione non è vuoto e se lunghezza Clique Candidata + lunghezza Intersezione > lunghezza Clique Massima.
Questo perché è inutile andare a visitare un sotto-grafo se si sa già che nel caso migliore non si raggiunge una grandezza della cricca maggiore di quella già trovata.

![Image of Pseudocode](https://i.imgur.com/yMS5Dii.png)

Figura 1 Pseudo-Codice Sequenziale dal documento utilizzato

## Algoritmo parallelo
### Primo Approccio

Il primo approccio usato è quello di parallelizzare la visita del grafo, ovvero in un grafo di N nodi si lanciano N threads ed ognuno esegue l’algoritmo sequenziale esposto precedentemente partendo da un nodo diverso.
L’approccio non ha prodotto buoni risultati.
Essendo un grafo randomico non si potevano predire gli accessi alla memoria che quindi avvenivano in modo caotico e il branch divergence era molto difficile da eliminare o minimizzare.
L’algoritmo lavorava sempre peggio del sequenziale.

### Secondo Approccio

Il secondo approccio che è stato utilizzato è stato quello di parallelizzare due punti dell’algoritmo:
1.	La creazione della lista Intersezione
2.	La copia di Clique Candidata in Clique Massima

Il secondo approccio è stato testato con due metodologie diverse:
•	Classico metodo dove si lancia il Kernel a seguito della copia della memoria HostToDevice e viceversa al termine dell’esecuzione
•	Utilizzando la libreria Thrust che mette a disposizione un metodo set_intersection che permette, appunto, di calcolare l’intersezione tra due liste.


### Senza libreria

Per il calcolo dell’intersezione vengono lanciati N blocchi con M Threads, dove N è la lunghezza della lista Intersezione e M la lunghezza della lista dei vicini del nodo attualmente visitato. Se N non è un multiplo di 32 allora viene lanciato un numero superiore di Thread, in modo da ottimizzare l’esecuzione.
Ogni thread esegue il controllo Intersezione[BlockIdx.x]==Vicini[ThreadIdx.x] e se tornava true allora il thread copia il valore in una lista per l’output (sul quale viene eseguita la memcpy DeviceToHost al termine dell’esecuzione del kernel).

Per la copia della cricca invece viene sfruttato l’uso di uno Stream. Prima del lancio dell’algoritmo viene fatta una cudaMalloc di Clique Massima e passata come parametro alla funzione ricorsiva.
Quando deve avvenire la copia, descritta nel punto 5 dell’algoritmo sequenziale, viene creato uno stream e viene fatta una cudaMemcpyAsync HostToDevice per copiare la nuova cricca all’interno di Clique Massima. Questo permette di fare la copia asincrona e permettere all’algoritmo di continuare l’esecuzione anche se la copia non è ancora finita.
Con questo metodo non viene lanciato un vero e proprio kernel, ma si sfrutta semplicemente la memoria del device e la possibilità di eseguire una copia asincrona. Nonostante l’algoritmo vada avanti la copia riesce sempre a terminare prima che se ne debba fare una nuova.
Al termine dell’algoritmo si farà una memcpy DeviceToHost di Clique Massima per estrarre la lista.

### Con libreria Thrust

Thrust è una libreria ufficiale Nvidia che permette una gestione più facile ed automatizzata del Kernel insieme a svariati metodi e algoritmi per la gestione o elaborazione delle liste/vettori.
Ha due componenti principali host_vector e device_vector. Il primo è un vector che risiede nella memoria host, quindi può interagire con le altre strutture dati del programma. Il secondo invece è un vector presente solo nella memoria device, quindi quando questo viene inizializzato e popolato è come se si facesse la cudaMalloc e MemCpy su un array.

Il procedimento è simile a quello già esposto: 
•	Per l’intersezione viene usato il metodo nativo di Thrust: set_intersection().
•	Per l’ottimizzazione della copia, invece, si crea un device_vector che viene passato alla funzione ricorsiva (come nel caso precedente, ma si faceva invece una cudaMalloc su un array) e nel momento che deve avvenire la copia della clique all’interno di Clique Massima viene usato il metodo copy().




## Risultati
Sono state fatte diverse misurazioni sul tempo di esecuzione in base alla dimensione e connettività del grafo.
Comparazione seriale/parallelo con l’approccio senza libreria (sull’ascissa il numero di nodi e sull’ordinata il tempo impiegato in minuti):
 
![Result 1](https://i.imgur.com/onm4wKr.png)

![Result 2](https://i.imgur.com/3m1xlYP.png) 


Il sequenziale non ha un valore corretto con più di 500 nodi con alta connettività, dal momento che il tempo impiegato superava le 12 ore, quindi è stato messo un valore simbolico.

Comparazione seriale/parallelo con Thrust(sull’ascissa il numero di nodi e sull’ordinata il tempo impiegato in minuti):
 ![Result 3](https://i.imgur.com/npSIJ18.png)

![Result 4](https://i.imgur.com/NzT52aP.png) 
 
Sono state eseguite misurazioni anche sulla percentuale utilizzo della memoria e sul Throughput medio degli SM, simulando un caso pessimo (alta connettività, quindi liste più lunghe e più threads lanciati) e un caso medio (connettività del grafo “media”). Sull’ascissa viene indicato il numero di nodi e sull’ordinata la % di utilizzo. 
Approccio senza libreria:

![Result 5](https://i.imgur.com/bCLMWXX.png?1)

![Result 6](https://i.imgur.com/sEiNzli.png) 
 
Approccio con Thrust:
 
![Result 7](https://i.imgur.com/172vISi.png)

![Result 8](https://i.imgur.com/OhfZjC5.png) 


Dai grafici si può notare che con Thrust i tempi di esecuzione tendono a degenerare leggermente di più, al crescere del numero di nodi del grafo, rispetto all’approccio senza libreria. Però si può notare anche che Thrust ottimizza molto di più l’utilizzo della memoria e il carico di lavoro medio degli SM. 

