---
title: Progetto visione artificiale
created: '2020-12-20T14:39:05.702Z'
modified: '2020-12-20T14:43:20.028Z'
---

# Progetto visione artificiale

- [ ] Leggere qualche paper e studiare lo stato dell'arte. (2 ore)

- [ ] Scegliere rete e dataset su cui è preaddestrata. (4 ore)

- [ ] Scegliere se e come cambiare l'architettura (se classificatore con 100 classi o un regressore). (6-8 ore)

- [ ] Preelaborare il dataset/labels forniteci (se tradurle in h5 oppure lasciarle così come sono), eventuali preprocessing, dobbiamo dividere in training/validation/test anche per identità, nel validation e nel test dobbiamo mettere entità diverse rispetto al training 70/15/15. (6-8 ore) shuffle dell'entità e delle immagini all'interno della cartella per ogni entità.
Dovremmo bilanciare per fasce d'età.(PROBABILMENTE NON è UTILE)

- [ ] Modificare il notebook colab aggiungendo il preprocessing e l'augmentation. (6-8 ore)

- [ ] Allenare il/i modelli e vedere le performance, usiamo il validation per modificare gli iperparametri della singola rete, mentre il test set per scegliere tra i vari modelli possibilmente senza reiterare su questo punto (overfitting).
