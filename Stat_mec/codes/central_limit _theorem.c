#include<stdlib.h>
#include<math.h>
#include<stdio.h>
#include<time.h>


double gaussdp(double g){
    g = exp(pow(-g,2));
    return g;
}

double gammadp(double m){
    if(m <= 0.0){
        return 0;
    }
    if(m > 0.0){
        m = tgamma(m);
    return m;
    }
}

double expdp(double e){
    if(e <= 0.0){
        return 0;
    }
    if(e > 0.0){
        e = exp(-e);
    return e;
    }
}

double uniformdp(double u){
//usando como parametros a = 0.10 e b = 0.75
    if(u < 0.10){
        return 0;
    }
    if(u > 0.75){
        return 1;
    }
    if(0.10 <= u && u <= 0.75){
        u = (u-0.1)/0.65;
    return u;
    }
}

int main(){
     int i,j,N=20,M=1000;
     double r,x,xm, Xm;
     FILE*random;
     random = fopen("fdp_gaussianaN20.txt","w+");
    
 //lembrar: calcular média de N num aleatorios para gerar M pontos da 
distribuição!
 
 
    srand(time(NULL));
    for(j=0;j<M;j++){
        xm = 0;
        for(j=0;j<N;j++){
            r = rand()/((double)RAND_MAX);
            x = (gaussdp(r));
            xm = xm + x;
        } xm = xm/N;
        Xm = Xm + xm;
        fprintf(random,"%d %lf\n", j, xm);
    } Xm = Xm/M;
    printf("a média e igual a: %lf", Xm);
    fclose(random);
    printf("seu arquivo foi criado com sucesso! :)\n");
return (0);
}