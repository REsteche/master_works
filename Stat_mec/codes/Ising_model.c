#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define N 30 //tamanho da rede -1 linha e -1 coluna que vão ser perididas nas iterações
#define Nt 784 //constante pra preencher as bordas periódicamente 
#define J 0.5 //constante ferromagnetismo(+/-); dividimos por 2 para evitar repeticao energias
#define B 0 
#define NE 2 //Vetor para guardar as energias do passo atual e posterior
#define T 5
#define C 10000 //Número de iterações

using namespace std;

void periodico (int M[N][N]){ //Tornando uma matriz quadrada periódica
     int i,j;
     for(j=1;j<(N-1);j++){
         M[0][j] = M[N-2][j];
         M[N-1][j] = M[1][j];
     }
     for(i=1;i<(N-1);i++){
         M[i][0] = M[i][N-2];
         M[i][N-1] = M[i][1];
     }
     M[0][0] = M[N-2][N-2];
     M[0][N-1] = M[N-2][1];
     M[N-1][0] = M[1][N-2];
     M[N-1][N-1] = M[1][1];
}

void energia (int M[N][N], double Energias[NE], double MEM[4], double t){ //Função onde ocorre as mudanças de energia
     int i, j, prob, prob2, de, O[N][N];
     double parametro, sorteio, beta, Eprov;
     for(i=0;i<N;i++){
         for(j=0;j<N;j++){
             O[i][j] = M[i][j];
         }
     }
     prob = rand()%(N-2) + 1; //escolhendo um spin aleatório da rede
     prob2 = rand()%(N-2) + 1;
     Energias[0] = 0;
     Energias[1] = 0;
     for(i=0;i<4;i++){
         MEM[i] = 0;
     }
    for(i=1;i<(N-1);i++){ //Calculando a energia atual
         for(j=1;j<(N-1);j++){
             Energias[0] += -J*((M[i][j]*M[i][j+1]) + (M[i][j]*M[i+1][j]) +
            (M[i][j]*M[i][j-1]) + (M[i][j]*M[i-1][j])) - B*M[i][j];
         }
     }
    O[prob][prob2] = -O[prob][prob2]; //Alterando o spin escolhido
     for(i=1;i<(N-1);i++){ //Calculando a nova energia
         for(j=1;j<(N-1);j++){
             Energias[1] += -J*((O[i][j]*O[i][j+1]) + (O[i][j]*O[i+1][j]) +
            (O[i][j]*O[i][j-1]) + (O[i][j]*O[i-1][j])) - B*O[i][j];
         }
     }
     de = Energias[1] - Energias[0]; //Diferença de energia entre os estados
     if(de>0){ //Analisando o caso onde a energia aumenta
         sorteio = rand()%RAND_MAX;
         sorteio = sorteio/RAND_MAX;
         parametro = exp(-de/t);
         if(sorteio<=parametro){
             M[prob][prob2] = O[prob][prob2];
         }
     }
     for(i=1;i<(N-1);i++){ //Iterando E, E^2, M e M^2
         for(j=1;j<(N-1);j++){
             Eprov = -J*((M[i][j]*M[i][j+1]) + (M[i][j]*M[i+1][j]) + (M[i][j]*M[i][j-1]) +
            (M[i][j]*M[i-1][j])) - B*M[i][j];
             MEM[0] += Eprov;
             MEM[1] += Eprov*Eprov;
             MEM[2] += M[i][j];
             MEM[3] += M[i][j]*M[i][j];
         }
     }
}

int main(){
     int i, j, M[N][N], k; //k é o contador de iterações
     double prob, t, Energias[NE], MEM[4]; //MEM guarda as grandezas desejadas (E, E^2, M e M^2)
     FILE* data = fopen("Dados.txt", "w+");
     srand((unsigned)time(NULL));
     for(t=0.5;t<=T;t+=0.1){ //Loop para analisar diferentes temperaturas
         for(i=0;i<NE;i++){ //Zerando os dados
             Energias[i] = 0;
         }
     for(i=0;i<4;i++){
         MEM[i] = 0;
     } 
    for(i=1;i<(N-1);i++){ //Definindo a população inicial
         for(j=1;j<(N-1);j++){
         /*prob = rand()%RAND_MAX;
         prob = prob/RAND_MAX;
     if(prob<=0.5){
         M[i][j] = 1; //isso daqui foi de quando estavamos gerando condição incial aleatoria, antes da dica do professor de começar todos em +/-1 
     } else {
         M[i][j] = -1;
        }*/
     M[i][j] = 1;
     }
 }
     periodico(M);
    for(k=0;k<C;k++){ // Loop das iterações
         energia(M,Energias,MEM,t);
         periodico(M);
     }
     //fprintf(data,"%g %g %g %g %g",t,MEM[0]/Nt,fabs(MEM[2])/Nt, 
    (float)((MEM[1]/Nt)-(MEM[0]*MEM[0])/(Nt*Nt))/(t*t), (float)((MEM[3]/Nt)-
    (MEM[2]*MEM[2])/(Nt*Nt))/t);
    //desistimos desse printf por conta de correlação
     fprintf(data,"%g %g ",t,MEM[0]/Nt);
     for(k=0;k<500;k++){ // Loop das iterações
         energia(M,Energias,MEM,t);
         periodico(M);
     }
     fprintf(data,"%g ",fabs(MEM[2])/Nt);
    for(k=0;k<500;k++){ // Loop das iterações
         energia(M,Energias,MEM,t);
         periodico(M);
    }
     fprintf(data,"%g ",(float)((MEM[1]/Nt)-(MEM[0]*MEM[0])/(Nt*Nt))/(t*t));
     for(k=0;k<500;k++){ // Loop das iterações
         energia(M,Energias,MEM,t);
         periodico(M);
     }
     fprintf(data,"%g",(float)((MEM[3]/Nt)-(MEM[2]*MEM[2])/(Nt*Nt))/t);
     fprintf(data,"\n");
     cout << t << " ok" << endl;
     }
     fclose(data);
}