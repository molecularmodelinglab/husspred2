$wnd.jsme.runAsyncCallback1('var D7="Assignment of aromatic double bonds failed",$pa="Smiles with leading parenthesis are not supported",aqa="SmilesParser: \'+\' found outside brackets",bqa="SmilesParser: closing bracket without opening one",cqa="SmilesParser: dangling open bond",dqa="SmilesParser: dangling ring closure",eqa="SmilesParser: nested square brackets found",fqa="SmilesParser: ring closure to same atom",gqa="SmilesParser: ringClosureAtom number out of range",hqa="SmilesParser: unexpected character found: \'",iqa="SmilesParser: unknown element label found";\nfunction E7(a,b){var c;c=a.A[b];return 3<=c&&4>=c||11<=c&&13>=c||19<=c&&31>=c||37<=c&&51>=c||55<=c&&84>=c||87<=c&&103>=c}function $(a,b){var c,d;c=b;for(d=0;0!=b;)0==a.c&&(a.e=(a.a[++a.d]&63)<<11,a.c=6),d|=(65536&a.e)>>16-c+b,a.e<<=1,--b,--a.c;return d}function F7(a,b,c){a.c=6;a.d=c;a.a=b;a.e=(b[a.d]&63)<<11}function G7(a,b){var c,d;c=~~(b/2);(d=a>=c)&&(a-=c);c=~~(b/32)*a/(c-a);return d?-c:c}function H7(){this.b=!0}w(24,1,{},H7);_.a=null;_.b=!1;_.c=0;_.d=0;_.e=0;_.f=null;\nfunction I7(a,b){var c,d,e;1==a.b.E[b]&&Gs(a.b,b,2);for(d=0;2>d;++d){c=D(a.b,d,b);Kv(a.b,c,!1);for(e=0;e<a.b.f[c];++e)a.a[Is(a.b,c,e)]=!1}}function J7(a){var b,c,d,e,f,g,h;do{h=!1;for(c=0;c<a.b.d;++c)if(a.a[c]){f=!1;for(e=0;2>e;++e){b=!1;d=D(a.b,e,c);for(g=0;g<a.b.f[d];++g)if(c!=Is(a.b,d,g)&&a.a[Is(a.b,d,g)]){b=!0;break}if(!b){f=!0;break}}f&&(h=!0,I7(a,c))}}while(h)}function K7(){}w(29,1,{},K7);_.a=null;_.b=null;\nfunction L7(a,b,c,d){a.b||(4==a.i||3==a.i&&-1!=a.c?a.b=!0:(a.j[a.i]=d,a.f[a.i]=b,a.k[a.i]=c,++a.i))}\nfunction jqa(a,b){var c,d,e,f;if(a.b)return 3;-1!=a.c&&(a.c=b[a.c]);for(e=0;e<a.i;++e)2147483647!=a.f[e]&&(a.f[e]=b[a.f[e]]);if(-1==a.c&&0==a.d){d=2147483647;f=-1;for(e=0;e<a.i;++e)d>a.k[e]&&(d=a.k[e],f=e);a.c=a.f[f];for(e=f+1;e<a.i;++e)a.f[e-1]=a.f[e],a.k[e-1]=a.k[e],a.j[e-1]=a.j[e];--a.i}f=(-1==a.c?0:1)+a.d+a.i;if(4<f||3>f)return 3;c=-1==a.c&&1==a.d||-1!=a.c&&cw(a.n.b,a.c);d=-1;for(e=0;e<a.i;++e)if(a.j[e]){if(-1!=d||c)return 3;d=e}f=!1;if(-1!=d)for(e=0;e<a.i;++e)!a.j[e]&&a.f[d]<a.f[e]&&(f=!f);d=\n!1;if(-1!=a.c&&!c)for(e=0;e<a.i;++e)a.c<a.f[e]&&(d=!d);e=a.f;c=a.k;var g,h,j;h=!1;for(g=1;g<a.i;++g)for(j=0;j<g;++j)e[j]>e[g]&&(h=!h),c[j]>c[g]&&(h=!h);return a.e^h^d^f?2:1}function M7(a,b,c,d,e,f){this.n=a;0!=d&&1!=d?this.b=!0:(this.a=b,this.c=c,this.d=d,this.e=f,this.i=0,this.j=C(Fs,Dr,-1,4,2),this.f=C(B,v,-1,4,1),this.k=C(B,v,-1,4,1),-1!=c&&1==d&&(L7(this,2147483647,e,!0),this.d=0))}w(30,1,{},M7);_.a=0;_.b=!1;_.c=0;_.d=0;_.e=!1;_.f=null;_.i=0;_.j=null;_.k=null;_.n=null;\nfunction N7(a){Es(a,15);if(a.b){var a=a.b,b;for(b=0;b<a.K.c;++b)if(0==(a.K.s[b]&67108864)&&3==a.V[b]){var c=a.K;c.s[b]|=67108864;c.N&=3}for(b=0;b<a.K.d;++b)3==a.k[b]&&2==Ms(a.K,b)&&Gs(a.K,b,26)}}function O7(){this.e=1}w(33,1,{},O7);\nfunction P7(a){var b,c;if(null==a||0==a.length||0==zw(a).length)return MZ(new JP,n,!0);c=new Mw;var d=new K7,e=QW(zw(a)),f,g,h,j,l,m,q,r,s,x,u,z,F,H,I,t,ea,da,U,ka,Fa,Dc,Qa,Bb,P,hb,ua,fa,ba,qa,Ea,Tb,S,Ia,cc,Xc,Ga;d.b=c;tv(d.b);Qa=null;j=C(B,v,-1,64,1);j[0]=-1;hb=C(B,v,-1,64,1);ua=C(B,v,-1,64,1);for(F=0;64>F;++F)hb[F]=-1;g=P=0;fa=Bb=qa=!1;m=0;ba=e.length;for(l=1;32>=e[P];)++P;for(;P<ba;)if(Ea=e[P++]&65535,Q7(Ea)||42==Ea){h=0;u=-1;H=Dc=I=!1;if(qa)82==Ea&&XO(e[P]&65535)?(da=null!=String.fromCharCode(e[P+\n1]&65535).match(/\\d/)?2:1,h=Wv(uv(e,P-1,1+da)),P+=da):(t=String.fromCharCode(e[P]&65535).toLowerCase().charCodeAt(0)==(e[P]&65535)&&Q7(e[P]&65535)?2:1,h=Wv(uv(e,P-1,t)),P+=t-1,u=0),64==e[P]&&(++P,64==e[P]&&(H=!0,++P),Dc=!0),72==e[P]&&(++P,u=1,XO(e[P]&65535)&&(u=e[P]-48,++P));else if(42==Ea)h=6,I=!0;else switch(String.fromCharCode(Ea).toUpperCase().charCodeAt(0)){case 66:P<ba&&114==e[P]?(h=35,++P):h=5;break;case 67:P<ba&&108==e[P]?(h=17,++P):h=6;break;case 70:h=9;break;case 73:h=53;break;case 78:h=\n7;break;case 79:h=8;break;case 80:h=15;break;case 83:h=16}if(0==h)throw new Dt(iqa);f=ov(d.b,h);I?(fa=!0,Nv(d.b,f,1)):Kv(d.b,f,String.fromCharCode(Ea).toLowerCase().charCodeAt(0)==Ea&&Q7(Ea));if(-1!=u&&1!=h){q=C(eu,Wr,-1,1,1);q[0]=u<<24>>24;var La=d.b,ab=f,Ec=q;null!=Ec&&0==Ec.length&&(Ec=null);null==Ec?null!=La.r&&(La.r[ab]=null):(null==La.r&&(La.r=C(mv,o,3,La.J,0)),La.r[ab]=Ec)}z=j[m];-1!=j[m]&&128!=l&&sv(d.b,f,j[m],l);l=1;j[m]=f;0!=g&&(Lv(d.b,f,g),g=0);(ka=!Qa?null:rw(Qa,nS(z)))&&L7(ka,f,P,1==\nh);Dc&&(!Qa&&(Qa=new Gw),Hw(Qa,nS(f),new M7(d,f,z,u,P,H)))}else if(46==Ea)l=128;else if(61==Ea)l=2;else if(35==Ea)l=4;else if(XO(Ea))if(U=Ea-48,qa){for(;P<ba&&XO(e[P]&65535);)U=10*U+e[P]-48,++P;g=U}else{Bb&&P<ba&&XO(e[P]&65535)&&(U=10*U+e[P]-48,++P);Bb=!1;if(64<=U)throw new Dt(gqa);if(-1==hb[U])hb[U]=j[m],ua[U]=P-1;else{if(hb[U]==j[m])throw new Dt(fqa);Qa&&((ka=rw(Qa,nS(hb[U])))&&L7(ka,j[m],ua[U],!1),(ka=rw(Qa,nS(j[m])))&&L7(ka,hb[U],P-1,!1));sv(d.b,j[m],hb[U],l);hb[U]=-1}l=1}else if(43==Ea){if(!qa)throw new Dt(aqa);\nfor(r=1;43==e[P];)++r,++P;1==r&&XO(e[P]&65535)&&(r=e[P]-48,++P);Ev(d.b,j[m],r)}else if(45==Ea){if(qa){for(r=-1;45==e[P];)--r,++P;-1==r&&XO(e[P]&65535)&&(r=48-e[P],++P);Ev(d.b,j[m],r)}}else if(40==Ea){if(-1==j[m])throw new Dt($pa);j[m+1]=j[m];++m}else if(41==Ea)--m;else if(91==Ea){if(qa)throw new Dt(eqa);qa=!0}else if(93==Ea){if(!qa)throw new Dt(bqa);qa=!1}else if(37==Ea)Bb=!0;else if(58==Ea)if(qa){for(ea=0;XO(e[P]&65535);)ea=10*ea+e[P]-48,++P;d.b.u[j[m]]=ea}else l=64;else if(47==Ea)l=17;else if(92==\nEa)l=9;else throw new Dt(hqa+String.fromCharCode(Ea)+Ce);if(1!=l)throw new Dt(cqa);for(F=0;64>F;++F)if(-1!=hb[F])throw new Dt(dqa);var va=d.b,bb,aa,ya,Y,Yb,G;G=C(B,v,-1,va.o,1);Y=C(Fs,Dr,-1,va.o,2);for(aa=0;aa<va.p;++aa)for(ya=0;2>ya;++ya)cw(va,va.B[ya][aa])&&!cw(va,va.B[1-ya][aa])&&(Y[va.B[ya][aa]]=!0);for(Yb=va.o-1;0<=Yb&&Y[Yb];)G[Yb]=Yb,--Yb;for(bb=0;bb<=Yb;++bb)if(Y[bb]){G[bb]=Yb;G[Yb]=bb;for(--Yb;0<=Yb&&Y[Yb];)G[Yb]=Yb,--Yb}else G[bb]=bb;d.b.M=!0;Es(d.b,1);for(f=0;f<d.b.o;++f)if(null!=(null==\nc.r?null:null==c.r[f]?null:uv(c.r[f],0,c.r[f].length))&&!Bv(d.b,f))if(x=(null==d.b.r?null:d.b.r[f])[0],d.b.A[f]<(zt(),lv).length&&null!=lv[d.b.A[f]]){s=!1;Tb=lu(d.b,f);Tb-=nu(d.b,f,Tb);for(Ia=lv[d.b.A[f]],cc=0,Xc=Ia.length;cc<Xc;++cc)if(S=Ia[cc],Tb<=S){s=!0;S!=Tb+x&&Dv(d.b,f,Tb+x);break}s||Dv(d.b,f,Tb+x)}var N,Ib,Ub,ld;for(N=0;N<d.b.c;++N)if(7==d.b.A[N]&&0==d.b.q[N]&&3<lu(d.b,N)&&0<d.b.k[N])for(ld=0;ld<d.b.f[N];++ld)if(Ib=Js(d.b,N,ld),Ub=Is(d.b,N,ld),1<Ms(d.b,Ub)&&Yv(d.b.A[Ib])){4==d.b.E[Ub]?Gs(d.b,\nUb,2):Gs(d.b,Ub,1);Ev(d.b,N,d.b.q[N]+1);Ev(d.b,Ib,d.b.q[Ib]-1);break}var yd,Zb,wa,tb,ma,dc,eb,T,ib,md,Oc,ub,Ra,mb,jb,Ic;Es(d.b,1);d.a=C(Fs,Dr,-1,d.b.d,2);for(wa=0;wa<d.b.d;++wa)64==d.b.E[wa]&&(Gs(d.b,wa,1),d.a[wa]=!0);Ic=new Hs(d.b,3);T=C(Fs,Dr,-1,Ic.i.c,2);for(Ra=0;Ra<Ic.i.c;++Ra){mb=Ps(Ic.i,Ra);T[Ra]=!0;for(eb=0;eb<mb.length;++eb)if(!Bv(d.b,mb[eb])){T[Ra]=!1;break}if(T[Ra]){jb=Ps(Ic.j,Ra);for(eb=0;eb<jb.length;++eb)d.a[jb[eb]]=!0}}for(wa=0;wa<d.b.d;++wa)if(!d.a[wa]&&0!=Ic.b[wa]&&Bv(d.b,D(d.b,0,\nwa))&&Bv(d.b,D(d.b,1,wa)))a:{var Ob=d,ec=wa,Fc=void 0,M=void 0,Ba=void 0,nd=void 0,vb=void 0,Cb=void 0,kb=void 0,wc=void 0,od=void 0,pd=void 0,dd=void 0,ja=void 0,xc=void 0,wc=C(B,v,-1,Ob.b.c,1),Cb=C(B,v,-1,Ob.b.c,1),kb=C(B,v,-1,Ob.b.c,1),od=C(B,v,-1,Ob.b.c,1),Fc=D(Ob.b,0,ec),M=D(Ob.b,1,ec);Cb[0]=Fc;Cb[1]=M;kb[0]=-1;kb[1]=ec;wc[Fc]=1;wc[M]=2;od[Fc]=-1;od[M]=Fc;for(pd=vb=1;vb<=pd&&15>wc[Cb[vb]];){xc=Cb[vb];for(dd=0;dd<Ob.b.f[xc];++dd)if(Ba=Js(Ob.b,xc,dd),Ba!=od[xc]){nd=Is(Ob.b,xc,dd);if(Ba==Fc){kb[0]=\nnd;for(ja=0;ja<=pd;++ja)Ob.a[kb[dd]]=!0;break a}Bv(Ob.b,Ba)&&0==wc[Ba]&&(++pd,Cb[pd]=Ba,kb[pd]=nd,wc[Ba]=wc[xc]+1,od[Ba]=xc)}++vb}}Es(d.b,3);for(Ra=0;Ra<Ic.i.c;++Ra)if(T[Ra]){mb=Ps(Ic.i,Ra);for(eb=0;eb<mb.length;++eb){var Jc;var Ca=d,fb=mb[eb],Gc=void 0;16==Ca.b.A[fb]&&0>=Ca.b.q[fb]||6==Ca.b.A[fb]&&0!=Ca.b.q[fb]||!Bv(Ca.b,fb)?Jc=!1:(Gc=null==su(Ca.b,fb)?0:(null==Ca.b.r?null:Ca.b.r[fb])[0],Jc=1>zv(Ca.b,fb)-lu(Ca.b,fb)-Gc||5!=Ca.b.A[fb]&&6!=Ca.b.A[fb]&&7!=Ca.b.A[fb]&&8!=Ca.b.A[fb]&&15!=Ca.b.A[fb]&&\n16!=Ca.b.A[fb]&&33!=Ca.b.A[fb]&&34!=Ca.b.A[fb]?!1:!0);if(!Jc){Kv(d.b,mb[eb],!1);for(md=0;md<d.b.f[mb[eb]];++md)d.a[Is(d.b,mb[eb],md)]=!1}}}J7(d);for(Ra=0;Ra<Ic.i.c;++Ra)if(T[Ra]&&6==Ps(Ic.j,Ra).length){jb=Ps(Ic.j,Ra);ib=!0;for(tb=0,ma=jb.length;tb<ma;++tb)if(wa=jb[tb],!d.a[wa]){ib=!1;break}ib&&(I7(d,jb[0]),I7(d,jb[2]),I7(d,jb[4]),J7(d))}for(ub=5;4<=ub;--ub){do{Oc=!1;for(wa=0;wa<d.b.d;++wa)if(d.a[wa]){for(eb=yd=0;2>eb;++eb){dc=D(d.b,eb,wa);for(md=0;md<d.b.f[dc];++md)d.a[Is(d.b,dc,md)]&&++yd}if(yd==\nub){I7(d,wa);J7(d);Oc=!0;break}}}while(Oc)}for(wa=0;wa<d.b.d;++wa)if(d.a[wa])throw new Dt(D7);for(Zb=0;Zb<d.b.c;++Zb)if(Bv(d.b,Zb))throw new Dt(D7);d.b.r=null;d.b.M=!1;var sc,Jb,Yc,Db,qd,Od,Pb,yc,zc,Ac,Zc;Es(d.b,3);zc=!1;Ac=C(B,v,-1,2,1);Zc=C(B,v,-1,2,1);yc=C(B,v,-1,2,1);for(Jb=0;Jb<d.b.d;++Jb)if(!ot(d.b,Jb)&&2==d.b.E[Jb]){for(Db=0;2>Db;++Db){Ac[Db]=-1;yc[Db]=-1;sc=D(d.b,Db,Jb);for(Pb=0;Pb<d.b.f[sc];++Pb)Yc=Is(d.b,sc,Pb),Yc!=Jb&&(17==d.b.E[Yc]||9==d.b.E[Yc]?(Ac[Db]=Js(d.b,sc,Pb),Zc[Db]=Yc):yc[Db]=\nJs(d.b,sc,Pb));if(-1==Ac[Db])break}if(-1!=Ac[0]&&-1!=Ac[1]){Od=d.b.E[Zc[0]]!=d.b.E[Zc[1]];qd=!1;for(Db=0;2>Db;++Db)-1!=yc[Db]&&yc[Db]<Ac[Db]&&(qd=!qd);Sv(d.b,Jb,Od^qd?2:1,!1);zc=!0}}for(Jb=0;Jb<d.b.d;++Jb)(17==d.b.E[Jb]||9==d.b.E[Jb])&&Gs(d.b,Jb,1);zc&&(d.b.N|=4);fx(new O7,d.b);if(Qa){for(Fa=R7((Ga=new PX(Qa),new S7(Qa,Ga)));jX(Fa.a.a);)ka=(Fa.a.b=Lu(Fa.a.a)).Oi(),Mv(d.b,ka.a,jqa(ka,G),!1);d.b.N|=4}dw(d.b);N7(d.b);fa&&Uv(d.b,!0);b=new lw(c);return BB(b.a.a)}\nfunction Q7(a){return null!=String.fromCharCode(a).match(/[A-Z]/i)}function R7(a){a=new SX(a.b.a);return new T7(a)}function S7(a,b){this.a=a;this.b=b}w(643,631,{},S7);_.Li=function(a){a:{var b,c;for(c=new SX((new PX(this.a)).a);jX(c.a);)if(b=c.b=Lu(c.a),b=b.Oi(),null==a?null==b:CA(a,b)){a=!0;break a}a=!1}return a};_.cf=function(){return R7(this)};_.xg=function(){return this.b.a.c};_.a=null;_.b=null;function T7(a){this.a=a}w(644,1,{},T7);_.ze=function(){return jX(this.a.a)};\n_.Ae=function(){return(this.a.b=Lu(this.a.a)).Oi()};_.Be=function(){RX(this.a)};_.a=null;function U7(){xX();this.a=6122;this.b=12230397}w(660,1,{},U7);w(713,603,es);\n_.de=function(){var a,b,c,d,e;a=b=d=null;if(this.b.a==(xS(),yS)&&this.b.i==(zS(),AS))try{var f=this.b.b,g,h,j;j=null;h=new Mw;vw(new Jw,h,new NN(new SN(f)))&&(g=new lw(h),j=BB(g.a.a));b=j;if(null==b)throw new Dt("V3000 read failed.");a=Kp;this.a.Mc.a="V3000 conversion provided by OpenChemLib"}catch(l){if(l=qt(l),E(l,101))c=l,d=c.be();else throw l;}else if(this.b.a==IY)try{var m=this.b.b,q,r,s,x,u,z,F;b=-1!=m.indexOf(fh)?(q=PW(m,fh),r=3<=q.length&&0<q[2].length,s=2<=q.length&&0<q[1].length,x=P7(q[0]),\nu=r?P7(q[2]):P7(n),z=s?P7(q[1]):P7(n),F=n,F+=ee,F+=qT(1,3)+qT(1,3),s&&(F+=qT(1,3)),F+=ga,F+=ce+x,F+=ce+u,s&&(F+=ce+z),F):P7(m);this.b.f==(uS(),DS)?a="readSMIRKS":this.b.f==FY&&(a="readSMILES");this.a.Mc.a="SMILES conversion provided by OpenChemLib"}catch(H){if(H=qt(H),E(H,101))c=H,d="SMILES parsing error:"+c.be();else throw H;}else if(d="Invalid or unsupported input",this.a.cd&&!this.b.d)try{var I,t=new H7,ea=zw(this.b.b),da;if(null==ea||0==ea.length)da=null;else{var U=QW(ea),ka,Fa,Dc,Qa,Bb;if(null==\nU)da=null;else{F7(t,U,0);ka=$(t,4);Qa=$(t,4);8<ka&&(ka=Qa);Fa=$(t,ka);Dc=$(t,Qa);Bb=new Aw(Fa,Dc);var P=null,hb,ua,fa,ba,qa,Ea,Tb,S,Ia,cc,Xc,Ga,La,ab,Ec,va,bb,aa,ya,Y,Yb,G,N,Ib,Ub,ld,yd,Zb,wa,tb,ma,dc,eb,T,ib,md,Oc,ub,Ra,mb,jb,Ic,Ob,ec,Fc,M,Ba,nd,vb,Cb,kb,wc,od,pd,dd,ja,xc,Jc,Ca,fb,Gc,sc,Jb,Yc,Db,qd,Od,Pb,yc,zc,Ac,Zc;qd=8;t.f=Bb;tv(t.f);if(!(null==U||0==U.length))if(null!=P&&0==P.length&&(P=null),F7(t,U,0),fa=$(t,4),va=$(t,4),8<fa&&(qd=fa,fa=va),0==fa)Uv(t.f,1==$(t,1));else{ba=$(t,fa);qa=$(t,va);\ndd=$(t,fa);Ca=$(t,fa);Jc=$(t,fa);Ub=$(t,fa);for(S=0;S<ba;++S)ov(t.f,6);for(M=0;M<dd;++M)rv(t.f,$(t,fa),7);for(M=0;M<Ca;++M)rv(t.f,$(t,fa),8);for(M=0;M<Jc;++M)rv(t.f,$(t,fa),$(t,8));for(M=0;M<Ub;++M)Ev(t.f,$(t,fa),$(t,4)-8);ld=1+qa-ba;eb=$(t,4);Ec=0;Ov(t.f,0,0);Pv(t.f,0,0);Qv(t.f,0,0);T=null!=P&&39<=P[0];Zc=zc=Pb=Db=0;tb=wa=!1;T&&(P.length>2*ba-2&&39==P[2*ba-2]||P.length>3*ba-3&&39==P[3*ba-3]?(tb=!0,Ba=(wa=P.length==3*ba-3+9)?3*ba-3:2*ba-2,ab=86*(P[Ba+1]-40)+P[Ba+2]-40,Db=Math.pow(10,ab/2E3-1),Ba+=\n2,Od=86*(P[Ba+1]-40)+P[Ba+2]-40,Pb=Math.pow(10,Od/1500-1),Ba+=2,yc=86*(P[Ba+1]-40)+P[Ba+2]-40,zc=Math.pow(10,yc/1500-1),wa&&(Ba+=2,Ac=86*(P[Ba+1]-40)+P[Ba+2]-40,Zc=Math.pow(10,Ac/1500-1))):wa=P.length==3*ba-3);t.b&&wa&&(P=null,T=!1);for(M=1;M<ba;++M)ib=$(t,eb),0==ib?(T&&(Ov(t.f,M,t.f.G[0].a+8*(P[2*M-2]-83)),Pv(t.f,M,t.f.G[0].b+8*(P[2*M-1]-83)),wa&&Qv(t.f,M,t.f.G[0].c+8*(P[2*ba-3+M]-83))),++ld):(Ec+=ib-1,T&&(Ov(t.f,M,Zs(t.f,Ec)+P[2*M-2]-83),Pv(t.f,M,$s(t.f,Ec)+P[2*M-1]-83),wa&&Qv(t.f,M,at(t.f,Ec)+\n(P[2*ba-3+M]-83))),sv(t.f,Ec,M,1));for(M=0;M<ld;++M)sv(t.f,$(t,fa),$(t,fa),1);vb=C(Fs,Dr,-1,qa,2);for(aa=0;aa<qa;++aa)switch(Yb=$(t,2),Yb){case 0:E7(t.f,D(t.f,0,aa))||E7(t.f,D(t.f,1,aa))?Gs(t.f,aa,32):vb[aa]=!0;break;case 2:Gs(t.f,aa,2);break;case 3:Gs(t.f,aa,4)}ua=$(t,fa);for(M=0;M<ua;++M)if(S=$(t,fa),8==qd)fb=$(t,2),3==fb?(Gv(t.f,S,1,0),Mv(t.f,S,1,!1)):Mv(t.f,S,fb,!1);else switch(fb=$(t,3),fb){case 4:Mv(t.f,S,1,!1);Gv(t.f,S,1,$(t,3));break;case 5:Mv(t.f,S,2,!1);Gv(t.f,S,1,$(t,3));break;case 6:Mv(t.f,\nS,1,!1);Gv(t.f,S,2,$(t,3));break;case 7:Mv(t.f,S,2,!1);Gv(t.f,S,2,$(t,3));break;default:Mv(t.f,S,fb,!1)}8==qd&&0==$(t,1)&&(t.f.I=!0);hb=$(t,va);for(M=0;M<hb;++M)if(aa=$(t,va),1==t.f.E[aa])switch(fb=$(t,3),fb){case 4:Sv(t.f,aa,1,!1);Rv(t.f,aa,1,$(t,3));break;case 5:Sv(t.f,aa,2,!1);Rv(t.f,aa,1,$(t,3));break;case 6:Sv(t.f,aa,1,!1);Rv(t.f,aa,2,$(t,3));break;case 7:Sv(t.f,aa,2,!1);Rv(t.f,aa,2,$(t,3));break;default:Sv(t.f,aa,fb,!1)}else Sv(t.f,aa,$(t,2),!1);Uv(t.f,1==$(t,1));Tb=null;for(xc=0;1==$(t,1);)switch(dc=\nxc+$(t,4),dc){case 0:ja=$(t,fa);for(M=0;M<ja;++M)S=$(t,fa),Nv(t.f,S,2048);break;case 1:ja=$(t,fa);for(M=0;M<ja;++M)S=$(t,fa),od=$(t,8),Lv(t.f,S,od);break;case 2:ja=$(t,va);for(M=0;M<ja;++M)aa=$(t,va),Gs(t.f,aa,64);break;case 3:ja=$(t,fa);for(M=0;M<ja;++M)S=$(t,fa),Nv(t.f,S,4096);break;case 4:ja=$(t,fa);for(M=0;M<ja;++M)S=$(t,fa),Yc=$(t,4)<<3,Nv(t.f,S,Yc);break;case 5:ja=$(t,fa);for(M=0;M<ja;++M)S=$(t,fa),Ea=$(t,2)<<1,Nv(t.f,S,Ea);break;case 6:ja=$(t,fa);for(M=0;M<ja;++M)S=$(t,fa),Nv(t.f,S,1);break;\ncase 7:ja=$(t,fa);for(M=0;M<ja;++M)S=$(t,fa),ec=$(t,4)<<7,Nv(t.f,S,ec);break;case 8:ja=$(t,fa);for(M=0;M<ja;++M){S=$(t,fa);Xc=$(t,4);Ia=C(B,v,-1,Xc,1);for(Cb=0;Cb<Xc;++Cb)cc=$(t,8),Ia[Cb]=cc;var fe=t.f,$c=S,nc=Ia;null==fe.t&&(fe.t=C(et,Mr,91,fe.J,0));null!=nc&&Lt(nc);fe.t[$c]=nc;fe.N=0;fe.H=!0}break;case 9:ja=$(t,va);for(M=0;M<ja;++M)aa=$(t,va),Yc=$(t,2)<<4,Tv(t.f,aa,Yc);break;case 10:ja=$(t,va);for(M=0;M<ja;++M)aa=$(t,va),G=$(t,4),Tv(t.f,aa,G);break;case 11:ja=$(t,fa);for(M=0;M<ja;++M)S=$(t,fa),\nNv(t.f,S,8192);break;case 12:ja=$(t,va);for(M=0;M<ja;++M)aa=$(t,va),N=$(t,8)<<6,Tv(t.f,aa,N);break;case 13:ja=$(t,fa);for(M=0;M<ja;++M)S=$(t,fa),Gc=$(t,3)<<14,Nv(t.f,S,Gc);break;case 14:ja=$(t,fa);for(M=0;M<ja;++M)S=$(t,fa),pd=$(t,5)<<17,Nv(t.f,S,pd);break;case 15:xc=16;break;case 16:ja=$(t,fa);for(M=0;M<ja;++M)S=$(t,fa),Jb=$(t,3)<<22,Nv(t.f,S,Jb);break;case 17:ja=$(t,fa);for(M=0;M<ja;++M)S=$(t,fa),Dv(t.f,S,$(t,4));break;case 18:ja=$(t,fa);wc=$(t,4);for(M=0;M<ja;++M){S=$(t,fa);ma=$(t,wc);kb=C(eu,\nWr,-1,ma,1);for(Cb=0;Cb<ma;++Cb)kb[Cb]=$(t,7)<<24>>24;var rd=t.f,Hc=S,nb=uv(kb,0,kb.length),Sc=void 0;if(null!=nb)if(0==nb.length)nb=null;else if(Sc=Wv(nb),0!=Sc&&R(nb,jv[Sc])||R(nb,gh))rv(rd,Hc,Sc),nb=null;null==nb?null!=rd.r&&(rd.r[Hc]=null):(null==rd.r&&(rd.r=C(mv,o,3,rd.J,0)),rd.r[Hc]=QW(nb))}break;case 19:ja=$(t,fa);for(M=0;M<ja;++M)S=$(t,fa),Ib=$(t,3)<<25,Nv(t.f,S,Ib);break;case 20:ja=$(t,va);for(M=0;M<ja;++M)aa=$(t,va),Jb=$(t,3)<<14,Tv(t.f,aa,Jb);break;case 21:ja=$(t,fa);for(M=0;M<ja;++M)S=\n$(t,fa),Iv(t.f,S,$(t,2)<<4);break;case 22:ja=$(t,fa);for(M=0;M<ja;++M)S=$(t,fa),Nv(t.f,S,268435456);break;case 23:ja=$(t,va);for(M=0;M<ja;++M)aa=$(t,va),Tv(t.f,aa,131072);break;case 24:ja=$(t,va);for(M=0;M<ja;++M)aa=$(t,va),Ea=$(t,2)<<18,Tv(t.f,aa,Ea);break;case 25:for(M=0;M<ba;++M)if(1==$(t,1)){var Pd=t.f;Pd.s[M]|=512}break;case 26:ja=$(t,va);Tb=C(B,v,-1,ja,1);for(M=0;M<ja;++M)Tb[M]=$(t,va);break;case 27:ja=$(t,fa);for(M=0;M<ja;++M)S=$(t,fa),Nv(t.f,S,536870912)}Ds(new Qs(t.f),vb);if(null!=Tb)for(ya=\n0,Y=Tb.length;ya<Y;++ya)aa=Tb[ya],Gs(t.f,aa,2==t.f.E[aa]?4:2);yd=0;if(null==P&&U.length>t.d+1&&(32==U[t.d+1]||9==U[t.d+1]))P=U,yd=t.d+2;if(null!=P)try{if(33==P[yd]||35==P[yd]){F7(t,P,yd+1);wa=1==$(t,1);tb=1==$(t,1);sc=2*$(t,4);bb=1<<sc;aa=0;for(S=1;S<ba;++S)aa<qa&&D(t.f,1,aa)==S?(Ic=D(t.f,0,aa++),jb=1):(Ic=0,jb=8),Ov(t.f,S,Zs(t.f,Ic)+jb*($(t,sc)-~~(bb/2))),Pv(t.f,S,$s(t.f,Ic)+jb*($(t,sc)-~~(bb/2))),wa&&Qv(t.f,S,at(t.f,Ic)+jb*($(t,sc)-~~(bb/2)));La=wa?1.5:(zt(),24);Ga=vv(t.f,ba,qa,La);if(35==P[yd]){Fc=\n0;Ob=C(B,v,-1,ba,1);for(S=0;S<ba;++S)Fc+=Ob[S]=Kt(t.f,S);for(S=0;S<ba;++S)for(M=0;M<Ob[S];++M)ec=ov(t.f,1),sv(t.f,S,ec,1),Ov(t.f,ec,Zs(t.f,S)+($(t,sc)-~~(bb/2))),Pv(t.f,ec,$s(t.f,S)+($(t,sc)-~~(bb/2))),wa&&Qv(t.f,ec,at(t.f,S)+($(t,sc)-~~(bb/2)));ba+=Fc}if(tb){var jc=$(t,sc),fc=Math.log(2E3)*Math.LOG10E*jc/(bb-1)-1;Db=Math.pow(10,fc);Pb=Db*G7($(t,sc),bb);zc=Db*G7($(t,sc),bb);wa&&(Zc=Db*G7($(t,sc),bb));jb=Db/Ga;for(S=0;S<ba;++S)Ov(t.f,S,Pb+jb*Zs(t.f,S)),Pv(t.f,S,zc+jb*$s(t.f,S)),wa&&Qv(t.f,S,Zc+jb*\nat(t.f,S))}else{jb=1.5/Ga;for(S=0;S<ba;++S)Ov(t.f,S,jb*Zs(t.f,S)),Pv(t.f,S,jb*$s(t.f,S)),wa&&Qv(t.f,S,jb*at(t.f,S))}}else if(wa&&!tb&&0==Db&&(Db=1.5),0!=Db&&0!=t.f.p){for(aa=Ga=0;aa<t.f.p;++aa)md=Zs(t.f,D(t.f,0,aa))-Zs(t.f,D(t.f,1,aa)),Oc=$s(t.f,D(t.f,0,aa))-$s(t.f,D(t.f,1,aa)),ub=wa?at(t.f,D(t.f,0,aa))-at(t.f,D(t.f,1,aa)):0,Ga+=Math.sqrt(md*md+Oc*Oc+ub*ub);Ga/=t.f.p;mb=Db/Ga;for(S=0;S<t.f.o;++S)Ov(t.f,S,Zs(t.f,S)*mb+Pb),Pv(t.f,S,$s(t.f,S)*mb+zc),wa&&Qv(t.f,S,at(t.f,S)*mb+Zc)}}catch(Sa){if(Sa=qt(Sa),\nE(Sa,101))Ra=Sa,Ra.be(),P=null,wa=!1;else throw Sa;}if((Zb=null!=P&&!wa)||t.b){Es(t.f,3);for(aa=0;aa<t.f.d;++aa)if(2==Ms(t.f,aa)&&!ot(t.f,aa)&&0==(t.f.C[aa]&3)){var hf=t.f;hf.C[aa]|=16777216}}!Zb&&t.b&&(t.f.N|=4,nd=new O7,nd.i=new U7,fx(nd,t.f),Zb=!0);Zb?(dw(t.f),N7(t.f)):wa||(t.f.N|=4)}da=Bb}}I=new lw(da);b=BB(I.a.a);a="readOCLCode";d=null}catch(Kc){if(Kc=qt(Kc),!E(Kc,101))throw Kc;}e=!1;if(null!=b&&null==d)try{(e=BS(this.a,b,!1))&&this.c&&qQ(this.a,a,0,0,0,!0)}catch(rb){if(rb=qt(rb),E(rb,101))d=\n"Invalid converted molfile";else throw rb;}this.a.jc=e;this.e?e?JS(this.e):KS(this.e,new Dt(d)):null!=d&&h4(this.a,d);this.d&&yL(this.a)};Z(643);Z(644);Z(24);Z(29);Z(30);V(N0)(1);\n//# sourceURL=1.js\n')
