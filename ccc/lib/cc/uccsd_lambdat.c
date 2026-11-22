/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Adapted from PySCF's pyscf/lib/cc/uccsd_t.c for Lambda-UCCSD(T).
 */

#include <stdio.h>
#include <stdlib.h>
#include "config.h"

void dgemm_(const char *, const char *, const int *, const int *, const int *,
            const double *, const double *, const int *, const double *,
            const int *, const double *, double *, const int *);

typedef struct {
        void *cache[6];
        short a;
        short b;
        short c;
        short padding;
} CacheJob;

static void make_permute_indices(int *idx, int n)
{
        const int nn = n * n;
        const int nnn = nn * n;
        int *idx0 = idx;
        int *idx1 = idx0 + nnn;
        int *idx2 = idx1 + nnn;
        int *idx3 = idx2 + nnn;
        int *idx4 = idx3 + nnn;
        int *idx5 = idx4 + nnn;
        int i, j, k, m;

        for (m = 0, i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
        for (k = 0; k < n; k++, m++) {
                idx0[m] = i * nn + j * n + k;
                idx1[m] = i * nn + k * n + j;
                idx2[m] = j * nn + i * n + k;
                idx3[m] = k * nn + i * n + j;
                idx4[m] = j * nn + k * n + i;
                idx5[m] = k * nn + j * n + i;
        } } }
}

static size_t gen_aaa_jobs(CacheJob *jobs, int nocc, int nvir,
                           int a0, int a1, int b0, int b1,
                           void *cache_row_a, void *cache_col_a,
                           void *cache_row_b, void *cache_col_b)
{
        const size_t nov = (size_t)nocc * (nocc + nvir) * sizeof(double);
        const int da = a1 - a0;
        const int db = b1 - b0;
        char *rowa = cache_row_a;
        char *cola = cache_col_a;
        char *rowb = cache_row_b;
        char *colb = cache_col_b;
        size_t m = 0;
        int a, b, c;

        if (b1 <= a0) {
                for (a = a0; a < a1; a++) {
                for (b = b0; b < b1; b++) {
                        for (c = 0; c < b0; c++, m++) {
                                jobs[m].a = a; jobs[m].b = b; jobs[m].c = c;
                                jobs[m].cache[0] = rowa + nov*(a1*(a-a0)+b);
                                jobs[m].cache[1] = rowa + nov*(a1*(a-a0)+c);
                                jobs[m].cache[2] = cola + nov*(da*b+a-a0);
                                jobs[m].cache[3] = rowb + nov*(b1*(b-b0)+c);
                                jobs[m].cache[4] = cola + nov*(da*c+a-a0);
                                jobs[m].cache[5] = colb + nov*(db*c+b-b0);
                        }
                        for (c = b0; c <= b; c++, m++) {
                                jobs[m].a = a; jobs[m].b = b; jobs[m].c = c;
                                jobs[m].cache[0] = rowa + nov*(a1*(a-a0)+b);
                                jobs[m].cache[1] = rowa + nov*(a1*(a-a0)+c);
                                jobs[m].cache[2] = cola + nov*(da*b+a-a0);
                                jobs[m].cache[3] = rowb + nov*(b1*(b-b0)+c);
                                jobs[m].cache[4] = cola + nov*(da*c+a-a0);
                                jobs[m].cache[5] = rowb + nov*(b1*(c-b0)+b);
                        }
                } }
        } else {
                for (a = a0; a < a1; a++) {
                for (b = a0; b <= a; b++) {
                        for (c = 0; c < a0; c++, m++) {
                                jobs[m].a = a; jobs[m].b = b; jobs[m].c = c;
                                jobs[m].cache[0] = rowa + nov*(a1*(a-a0)+b);
                                jobs[m].cache[1] = rowa + nov*(a1*(a-a0)+c);
                                jobs[m].cache[2] = rowa + nov*(a1*(b-a0)+a);
                                jobs[m].cache[3] = rowa + nov*(a1*(b-a0)+c);
                                jobs[m].cache[4] = cola + nov*(da*c+a-a0);
                                jobs[m].cache[5] = cola + nov*(da*c+b-a0);
                        }
                        for (c = a0; c <= b; c++, m++) {
                                jobs[m].a = a; jobs[m].b = b; jobs[m].c = c;
                                jobs[m].cache[0] = rowa + nov*(a1*(a-a0)+b);
                                jobs[m].cache[1] = rowa + nov*(a1*(a-a0)+c);
                                jobs[m].cache[2] = rowa + nov*(a1*(b-a0)+a);
                                jobs[m].cache[3] = rowa + nov*(a1*(b-a0)+c);
                                jobs[m].cache[4] = rowa + nov*(a1*(c-a0)+a);
                                jobs[m].cache[5] = rowa + nov*(a1*(c-a0)+b);
                        }
                } }
        }
        return m;
}

/* Add W and V for one virtual permutation.  w and v may alias. */
static void get_wv_aaa(double *w, double *v, double *scratch,
                       const double *fvohalf, const double *vooo,
                       const double *vvop, const double *t1T,
                       const double *t2T, int nocc, int nvir,
                       int a, int b, int c, const int *idx)
{
        const double zero = 0, one = 1, minus_one = -1;
        const char trans_n = 'N', trans_t = 'T';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const size_t nooo = (size_t)nocc * noo;
        const size_t nvoo = (size_t)nvir * noo;
        const double *pt2;
        int i, j, k, n;

        dgemm_(&trans_n, &trans_n, &noo, &nocc, &nvir,
               &minus_one, t2T + c*nvoo, &noo, vvop + nocc, &nmo,
               &zero, scratch, &noo);
        dgemm_(&trans_n, &trans_t, &nocc, &noo, &nocc,
               &minus_one, t2T + b*nvoo + c*noo, &nocc,
               vooo + a*nooo, &noo, &one, scratch, &nocc);

        pt2 = t2T + a*nvoo + b*noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                w[idx[n]] += scratch[n];
                v[idx[n]] += vvop[i*nmo+j] * t1T[c*nocc+k]
                           + pt2[i*nocc+j] * fvohalf[c*nocc+k];
        } } }
}

static void permute_aaa(double *out, const double *in, int n)
{
        const int nn = n*n;
        int i, j, k;
        for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
        for (k = 0; k < n; k++) {
                out[i*nn+j*n+k] = in[i*nn+j*n+k] + in[j*nn+k*n+i]
                                  + in[k*nn+i*n+j] - in[k*nn+j*n+i]
                                  - in[i*nn+k*n+j] - in[j*nn+i*n+k];
        } } }
}

static double energy_aaa(const double *w, const double *z,
                         const double *mo_energy, int nocc,
                         int a, int b, int c, double fac)
{
        const double abc = mo_energy[nocc+a] + mo_energy[nocc+b]
                         + mo_energy[nocc+c];
        double e = 0;
        int i, j, k, n;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                e += fac * w[n] * z[n]
                   / (mo_energy[i] + mo_energy[j] + mo_energy[k] - abc);
        } } }
        return e;
}

static double contract_aaa(int nocc, int nvir, int a, int b, int c,
                           const double *mo_energy,
                           const double *t1T, const double *t2T,
                           const double *l1T, const double *l2T,
                           const double *fvohalf, const double *vooo,
                           double *work, void **cache, const int *indices)
{
        const int n3 = nocc*nocc*nocc;
        double *wr = work;
        double *left = wr + n3;
        double *z = left + n3;
        int i;
        for (i = 0; i < n3; i++) wr[i] = left[i] = 0;

#define BUILD_AAA(W,V,S1,S2) \
        get_wv_aaa(W,V,z,fvohalf,vooo,cache[0],S1,S2,nocc,nvir,a,b,c,indices); \
        get_wv_aaa(W,V,z,fvohalf,vooo,cache[1],S1,S2,nocc,nvir,a,c,b,indices+n3); \
        get_wv_aaa(W,V,z,fvohalf,vooo,cache[2],S1,S2,nocc,nvir,b,a,c,indices+2*n3); \
        get_wv_aaa(W,V,z,fvohalf,vooo,cache[3],S1,S2,nocc,nvir,b,c,a,indices+3*n3); \
        get_wv_aaa(W,V,z,fvohalf,vooo,cache[4],S1,S2,nocc,nvir,c,a,b,indices+4*n3); \
        get_wv_aaa(W,V,z,fvohalf,vooo,cache[5],S1,S2,nocc,nvir,c,b,a,indices+5*n3)

        BUILD_AAA(wr, left, t1T, t2T);
        for (i = 0; i < n3; i++) left[i] = 0;
        BUILD_AAA(left, left, l1T, l2T);
#undef BUILD_AAA
        permute_aaa(z, left, nocc);

        if (a == c) return energy_aaa(wr, z, mo_energy, nocc, a,b,c, 1./6);
        if (a == b || b == c)
                return energy_aaa(wr, z, mo_energy, nocc, a,b,c, .5);
        return energy_aaa(wr, z, mo_energy, nocc, a,b,c, 1.);
}

void CCuccsd_lambdat_aaa(double *e_tot, double *mo_energy,
                         double *t1T, double *t2T,
                         double *l1T, double *l2T,
                         double *vooo, double *fvo,
                         int nocc, int nvir, int a0, int a1, int b0, int b1,
                         void *cache_row_a, void *cache_col_a,
                         void *cache_row_b, void *cache_col_b)
{
        const int da = a1-a0, db = b1-b0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * (size_t)da*db*b1);
        size_t njobs = gen_aaa_jobs(jobs, nocc, nvir, a0,a1,b0,b1,
                                    cache_row_a,cache_col_a,
                                    cache_row_b,cache_col_b);
        double *fvohalf = malloc(sizeof(double) * (size_t)nvir*nocc);
        int *indices = malloc(sizeof(int) * (size_t)nocc*nocc*nocc*6);
        int i;
        for (i = 0; i < nvir*nocc; i++) fvohalf[i] = .5*fvo[i];
        make_permute_indices(indices, nocc);

#pragma omp parallel default(none) shared(njobs,nocc,nvir,mo_energy,t1T,t2T,l1T,l2T,vooo,fvohalf,jobs,e_tot,indices,stderr)
{
        double *work = malloc(sizeof(double) * ((size_t)nocc*nocc*nocc*3+2));
        double e = 0;
        size_t job;
        if (work == NULL) {
                fprintf(stderr, "malloc failed in CCuccsd_lambdat_aaa\n");
                exit(1);
        }
#pragma omp for schedule(dynamic,4)
        for (job = 0; job < njobs; job++) {
                e += contract_aaa(nocc,nvir,jobs[job].a,jobs[job].b,jobs[job].c,
                                  mo_energy,t1T,t2T,l1T,l2T,fvohalf,vooo,
                                  work,jobs[job].cache,indices);
        }
        free(work);
#pragma omp critical
        *e_tot += e;
}
        free(indices);
        free(fvohalf);
        free(jobs);
}

typedef struct {
        double *mo_ea, *mo_eb, *fvo, *fVO;
        double *vooo, *vOoO, *VoOo;
        double *t1aT, *t1bT, *t2aaT, *t2abT;
} BaaData;

static void get_wv_baa(double *w, double *v, const BaaData *d,
                       double **cache, int nocca, int noccb,
                       int nvira, int nvirb, int a, int b, int c,
                       double *small)
{
        double *vvop=cache[0], *vVoP=cache[1], *VvOp=cache[2];
        const double zero=0, one=1, two=2;
        const char trans_t='T', trans_n='N';
        const int nmoa=nocca+nvira, nmob=noccb+nvirb;
        const int noo=nocca*nocca, nOo=noccb*nocca;
        const size_t nooo=(size_t)nocca*noo, noOo=(size_t)nocca*nOo;
        const size_t nOoO=(size_t)noccb*nOo, nvoo=(size_t)nvira*noo;
        const int nVoO=nvirb*nOo;
        double *t1a2=small, *fvo2=t1a2+nocca, *fVOhalf=fvo2+nocca;
        double *pt2aa, *pt2ab;
        int i,j,k,n;

        dgemm_(&trans_t,&trans_t,&nocca,&nOo,&nvira,&two,
               vvop+nocca,&nmoa,d->t2abT+a*nOo,&nVoO,&zero,v,&nocca);
        dgemm_(&trans_t,&trans_t,&nocca,&nOo,&nvirb,&two,
               vVoP+noccb,&nmob,d->t2abT+b*(size_t)nVoO,&nOo,&one,v,&nocca);
        dgemm_(&trans_n,&trans_t,&nOo,&nocca,&nocca,&one,
               d->VoOo+a*noOo,&nOo,d->t2aaT+b*nvoo+c*noo,&nocca,
               &one,v,&nOo);
        dgemm_(&trans_t,&trans_t,&nocca,&nOo,&noccb,&two,
               d->t2abT+b*(size_t)nVoO+a*nOo,&noccb,
               d->vOoO+c*nOoO,&nOo,&one,v,&nocca);
        for (n=0,i=0; i<noccb; i++) {
        for (j=0; j<nocca; j++) {
        for (k=0; k<nocca; k++,n++) w[n]=v[j*nOo+i*nocca+k];
        } }
        dgemm_(&trans_n,&trans_n,&noo,&noccb,&nvira,&one,
               d->t2aaT+b*nvoo,&noo,VvOp+nocca,&nmoa,&one,w,&noo);
        dgemm_(&trans_n,&trans_t,&noo,&noccb,&nocca,&two,
               d->vooo+c*nooo,&noo,
               d->t2abT+b*(size_t)nVoO+a*nOo,&noccb,&one,w,&noo);

        for (i=0; i<nocca; i++) {
                t1a2[i]=2*d->t1aT[b*nocca+i];
                fvo2[i]=2*d->fvo[b*nocca+i];
        }
        for (i=0; i<noccb; i++) fVOhalf[i]=.5*d->fVO[a*noccb+i];
        pt2aa=d->t2aaT+b*nvoo+c*noo;
        pt2ab=d->t2abT+(c*nvirb+a)*(size_t)nOo;
        for (n=0,i=0; i<noccb; i++) {
        for (j=0; j<nocca; j++) {
        for (k=0; k<nocca; k++,n++) {
                v[n]=w[n]+vvop[k*nmoa+j]*d->t1bT[a*noccb+i]
                     +VvOp[i*nmoa+k]*t1a2[j]
                     +pt2aa[j*nocca+k]*fVOhalf[i]
                     +pt2ab[k*noccb+i]*fvo2[j];
        } } }
}

static void permute_baa(double *out, const double *w, int nocca, int noccb)
{
        const int noo=nocca*nocca;
        int i,j,k,n;
        for (n=0,i=0; i<noccb; i++) {
        for (j=0; j<nocca; j++) {
        for (k=0; k<nocca; k++,n++)
                out[n]=w[i*noo+j*nocca+k]-w[i*noo+k*nocca+j];
        } }
}

static double energy_baa(const double *z0,const double *z1,
                         const double *w0,const double *w1,
                         const double *ea,const double *eb,
                         int nocca,int noccb,int a,int b,int c,double fac)
{
        const int noo=nocca*nocca;
        const double abc=eb[noccb+a]+ea[nocca+b]+ea[nocca+c];
        double e=0;
        int i,j,k;
        for (i=0;i<noccb;i++) {
        for (j=0;j<nocca;j++) {
        for (k=0;k<nocca;k++) {
                e+=(z0[i*noo+j*nocca+k]+z1[i*noo+k*nocca+j])
                  *(w0[i*noo+j*nocca+k]+w1[i*noo+k*nocca+j])*fac
                  /(eb[i]+ea[j]+ea[k]-abc);
        } } }
        return e;
}

static double contract_baa(int nocca,int noccb,int nvira,int nvirb,
                           int a,int b,int c,const BaaData *right,
                           const BaaData *left,void **cache,double *work)
{
        const int n3=noccb*nocca*nocca;
        double *wr0=work, *wr1=wr0+n3, *vl0=wr1+n3;
        double *vl1=vl0+n3, *tmp=vl1+n3;
        double *small=tmp+n3;
        get_wv_baa(wr0,vl0,right,(double **)cache,nocca,noccb,nvira,nvirb,
                   a,b,c,small);
        get_wv_baa(wr1,vl1,right,((double **)cache)+3,nocca,noccb,nvira,nvirb,
                   a,c,b,small);
        get_wv_baa(tmp,vl0,left,(double **)cache,nocca,noccb,nvira,nvirb,
                   a,b,c,small);
        get_wv_baa(tmp,vl1,left,((double **)cache)+3,nocca,noccb,nvira,nvirb,
                   a,c,b,small);
        permute_baa(tmp,vl0,nocca,noccb);
        permute_baa(vl0,vl1,nocca,noccb);
        return energy_baa(tmp,vl0,wr0,wr1,right->mo_ea,right->mo_eb,
                          nocca,noccb,a,b,c,b==c?.5:1.);
}

static size_t gen_baa_jobs(CacheJob *jobs,int nocca,int noccb,
                           int nvira,int nvirb,int a0,int a1,int b0,int b1,
                           void *cache_row_a,void *cache_col_a,
                           void *cache_row_b,void *cache_col_b)
{
        const size_t nov=(size_t)nocca*(nocca+nvira)*sizeof(double);
        const size_t noV=(size_t)nocca*(noccb+nvirb)*sizeof(double);
        const size_t nOv=(size_t)noccb*(nocca+nvira)*sizeof(double);
        const int da=a1-a0, db=b1-b0;
        char *rowa=cache_row_a,*cola=cache_col_a,*rowb=cache_row_b,*colb=cache_col_b;
        size_t m=0;
        int a,b,c;
        for (a=a0;a<a1;a++) {
        for (b=b0;b<b1;b++) {
        for (c=0;c<=b;c++,m++) {
                jobs[m].a=a; jobs[m].b=b; jobs[m].c=c;
                jobs[m].cache[0]=(c<b0?colb+nov*(db*c+b-b0):rowb+nov*(b1*(c-b0)+b));
                jobs[m].cache[1]=cola+noV*(da*c+a-a0);
                jobs[m].cache[2]=rowa+nOv*(nvira*(a-a0)+c);
                jobs[m].cache[3]=rowb+nov*(b1*(b-b0)+c);
                jobs[m].cache[4]=cola+noV*(da*b+a-a0);
                jobs[m].cache[5]=rowa+nOv*(nvira*(a-a0)+b);
        } } }
        return m;
}

void CCuccsd_lambdat_baa(double *e_tot,double *mo_ea,double *mo_eb,
                         double *t1aT,double *t1bT,double *t2aaT,double *t2abT,
                         double *l1aT,double *l1bT,double *l2aaT,double *l2abT,
                         double *vooo,double *vOoO,double *VoOo,
                         double *fvo,double *fVO,
                         int nocca,int noccb,int nvira,int nvirb,
                         int a0,int a1,int b0,int b1,
                         void *cache_row_a,void *cache_col_a,
                         void *cache_row_b,void *cache_col_b)
{
        const int da=a1-a0, db=b1-b0;
        CacheJob *jobs=malloc(sizeof(CacheJob)*(size_t)da*db*b1);
        size_t njobs=gen_baa_jobs(jobs,nocca,noccb,nvira,nvirb,a0,a1,b0,b1,
                                  cache_row_a,cache_col_a,cache_row_b,cache_col_b);
        BaaData right={mo_ea,mo_eb,fvo,fVO,vooo,vOoO,VoOo,t1aT,t1bT,t2aaT,t2abT};
        BaaData left ={mo_ea,mo_eb,fvo,fVO,vooo,vOoO,VoOo,l1aT,l1bT,l2aaT,l2abT};

#pragma omp parallel default(none) shared(njobs,nocca,noccb,nvira,nvirb,right,left,jobs,e_tot,stderr)
{
        const size_t nwork=(size_t)noccb*nocca*nocca*5+nocca*2+noccb+2;
        double *work=malloc(sizeof(double)*nwork);
        double e=0;
        size_t job;
        if (work==NULL) {
                fprintf(stderr,"malloc failed in CCuccsd_lambdat_baa\n");
                exit(1);
        }
#pragma omp for schedule(dynamic,4)
        for (job=0;job<njobs;job++)
                e+=contract_baa(nocca,noccb,nvira,nvirb,
                                jobs[job].a,jobs[job].b,jobs[job].c,
                                &right,&left,jobs[job].cache,work);
        free(work);
#pragma omp critical
        *e_tot+=e;
}
        free(jobs);
}
