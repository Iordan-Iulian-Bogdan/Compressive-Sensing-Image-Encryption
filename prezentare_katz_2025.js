// Generează prezentarea în română pentru Katz et al., CVPRW 2025.
// Dependență: pptxgenjs (instalat local în directorul temporar de lucru).
const pptxgen = require('C:/Users/bogdani/AppData/Local/Temp/opencode/paper-slides/node_modules/pptxgenjs');
const path = require('path');
const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'Prezentare academică';
pptx.subject = 'Katz et al. — Rethinking Compressive Sensing, CVPRW 2025';
pptx.title = 'Eșantionare compresivă pentru super-rezoluția video';
pptx.lang = 'ro-RO';
pptx.theme = { headFontFace: 'Aptos Display', bodyFontFace: 'Aptos', lang: 'ro-RO' };

const C = { bg:'0B1220', panel:'17253B', panel2:'1C304A', white:'F4F7FB', muted:'B9C9DB', cyan:'52D6D3', amber:'FFC870', green:'80E2B0', red:'FF9C95', line:'34475D' };
const SW=13.333, SH=7.5;
const rect=(s,x,y,w,h,fill=C.panel,r=0.16,line=fill)=>s.addShape(r?pptx.ShapeType.roundRect:pptx.ShapeType.rect,{x,y,w,h,rectRadius:r, radius:r,line:{color:line,width:0.8},fill:{color:fill}});
function txt(s,t,x,y,w,h,size=22,color=C.white,opts={}){s.addText(t,{x,y,w,h,fontFace:'Aptos',fontSize:size,color,bold:false,breakLine:false,margin:0,vertAlign:'mid',...opts});}
function title(s,t,sub){txt(s,t,0.72,0.44,12.0,0.55,30,C.white,{bold:true});if(sub)txt(s,sub,0.75,1.04,11.9,0.38,13,C.muted);}
function base(heading,sub){const s=pptx.addSlide();s.background={color:C.bg};s.addShape(pptx.ShapeType.line,{x:0.73,y:1.51,w:11.87,h:0,line:{color:C.line,width:1}});if(heading)title(s,heading,sub);txt(s,'KATZ ET AL.  •  CVPRW 2025',0.72,7.15,6,0.19,9,C.muted);txt(s,String(pptx._slides.length).padStart(2,'0'),12.04,7.10,0.52,0.25,10,C.cyan,{align:'right',bold:true});return s;}
function card(s,x,y,w,h,head,body,accent=C.cyan){rect(s,x,y,w,h);rect(s,x,y,0.055,h,accent);txt(s,head,x+0.22,y+0.17,w-0.42,0.44,20,accent,{bold:true});txt(s,body,x+0.22,y+0.74,w-0.44,h-0.85,16,C.white,{breakLine:false,vertAlign:'top'});}
function tag(s,t,x,y,w,color=C.cyan){rect(s,x,y,w,0.37,C.panel2,0.12);txt(s,t,x+0.11,y+0.02,w-0.22,0.32,13,color,{bold:true,align:'center'});}
function notes(s,text){s.addNotes(text);}
function mask(s,grid,x,y,side,primary=C.cyan){let n=grid.length,gap=0.024,cell=(side-(n-1)*gap)/n;for(let r=0;r<n;r++)for(let c=0;c<n;c++){let val=grid[r][c];rect(s,x+c*(cell+gap),y+r*(cell+gap),cell,cell,val?primary:C.panel2,0.02,val?primary:C.line);}}
function bullet(s,lines,x,y,w,{size=20,gap=0.84,color=C.white}={}){lines.forEach((line,i)=>{rect(s,x,y+i*gap+0.19,0.09,0.09,C.cyan);txt(s,line,x+0.25,y+i*gap,w-0.25,gap-0.06,size,color,{vertAlign:'mid'});});}
function arrow(s,x,y,w=0.54,color=C.cyan){s.addShape(pptx.ShapeType.chevron,{x,y,w,h:0.42,line:{color},fill:{color}});}

// 1 — titlu
{
 let s=base();tag(s,'PREZENTARE  •  ~12–14 MIN',0.75,0.72,3.1);txt(s,'Când senzorul alege\nce informație păstrează',0.75,1.56,11.9,1.65,39,C.white,{bold:true,breakLine:false});
 txt(s,'Eșantionare compresivă pentru super-rezoluția video',0.78,3.43,10.9,0.6,24,C.cyan);
 rect(s,0.76,4.48,11.8,1.13,C.panel);txt(s,'Ruthy Katz, Adi Teitel, Moran Mordechay, Adi Falik, Eli Bery, Maya Mayberg',1.04,4.7,11.25,0.3,18,C.white);txt(s,'„Rethinking Compressive Sensing: A Compression Framework for Video Super-Resolution” • CVPR Workshops 2025',1.04,5.08,11.2,0.31,14,C.muted);
 notes(s,'Timp: aproximativ 35 de secunde. Articolul propune schimbarea modului în care senzorul comprimă video-ul, astfel încât reconstrucția ulterioară să primească informație mai utilă. Voi începe cu noțiunea de eșantionare compresivă și voi ajunge la schema propusă, apoi la rezultatele experimentale.');
}
// 2 — motivație
{
 let s=base('De ce contează achiziția?', 'Un senzor cu foarte mulți pixeli nu înseamnă automat un video stocat la aceeași rezoluție.');
 card(s,0.78,1.83,3.77,3.65,'Senzor','Până la ~200 MP în telefoanele moderne; pixeli mici și volum mare de date.',C.cyan);
 card(s,4.79,1.83,3.76,3.65,'Compresie pe senzor','Pixel binning: se adună semnale din mai mulți pixeli pentru SNR și trafic de date acceptabile.',C.amber);
 card(s,8.80,1.83,3.75,3.65,'Video rezultat','Super-rezoluția încearcă să recupereze detaliile, dar binningul poate pierde ireversibil informație.',C.green);
 txt(s,'Întrebarea articolului: putem comprima mai inteligent, fără a crește rezoluția fluxului video?',1.01,5.95,11.4,0.55,22,C.white,{bold:true,align:'center'});
 notes(s,'Timp: 55 de secunde. Senzorii de telefon pot avea aproximativ 200 de megapixeli, dar rezoluția obișnuită pentru filmare este mult mai mică. Binningul adună charge din grupuri de pixeli și ajută atât sensibilitatea, cât și lățimea de bandă. Dacă însă am însumat aceiași pixeli în fiecare cadru, informația spațială fină eliminată nu mai apare în date. Ideea este să proiectăm compresia împreună cu reconstrucția.');
}
// 3 — CS intro
{
 let s=base('Introducere: ce este compressive sensing?', 'Eșantionăm mai puțin decât numărul necunoscutelor, folosind structura semnalului.');
 rect(s,0.83,1.9,11.67,1.12,C.panel2);txt(s,'y = Φx + η',1.15,2.11,3.9,0.58,37,C.cyan,{bold:true});txt(s,'x: semnalul  •  Φ: măsurarea  •  y: datele achiziționate  •  η: zgomotul',5.02,2.23,7.10,0.42,17,C.white);
 bullet(s,['Dacă x este rar sau compresibil într-o bază Ψ, îl putem aproxima din M < N măsurători.','Măsurările trebuie să păstreze informație diversă; apoi un algoritm reconstruiește x.','În video, cadrele vecine adaugă redundanță și indicii complementare.'],1.1,3.43,11.2,{size:21,gap:0.94});
 txt(s,'CS clasic: reconstruim un semnal din proiecții.  Aici: proiectăm proiecțiile pentru un VSR util.',1.05,6.49,11.1,0.42,17,C.amber,{align:'center'});
 notes(s,'Timp: 75 de secunde. Compressive sensing, sau eșantionarea compresivă, este ideea de a face măsurări liniare mai puține decât numărul de valori din semnal, și totuși de a-l reconstrui aproximativ dacă are structură, de exemplu este rar într-un dicționar. Litera Phi reprezintă modul în care măsurăm. Contează nu doar cât măsurăm, ci și câtă informație independentă aduc măsurările. În acest articol reconstrucția este învățată de o rețea de super-rezoluție video; nu se rezolvă explicit un program clasic de minimizare L1.');
}
// 4 — bridge
{
 let s=base('Din CS clasic la video', 'Schimbarea-cheie: măști diferite pe cadre consecutive.');
 card(s,0.82,1.88,5.57,3.95,'CS clasic','Mai multe măsurători ale aceluiași semnal static.\n\nÎntr-un senzor video, recitirea repetată a aceluiași cadru ar costa timp și ar reduce rata de cadre.',C.amber);
 card(s,6.69,1.88,5.81,3.95,'Propunerea Katz et al.','Fiecare cadru este măsurat cu o mască; măștile se schimbă între cadre.\n\nCadrele apropiate sunt asemănătoare, iar diferențele mici pot fi tratate aproximativ drept zgomot.',C.cyan);
 txt(s,'Rezultatul: informație sub-pixel complementară chiar și în zone cu puțină mișcare.',1.09,6.08,11.2,0.56,21,C.green,{bold:true,align:'center'});
 notes(s,'Timp: 60 de secunde. În teoria clasică luăm diverse proiecții ale aceluiași obiect. Camera trebuie însă să livreze cadre la ritmul ei, nu să citească același cadru de multe ori. Autorii înlocuiesc citirile repetate cu măsurări în cadre succesive, folosind asemănarea temporală. Nu este o identitate exactă pentru scene care se schimbă rapid; este o aproximație exploatată de modelul VSR. Acest lucru este deosebit de util pentru zone statice, unde mișcarea nu furnizează singură decalări sub-pixel.');
}
// 5 — pipeline
{
 let s=base('Fluxul propus: comprimare → reconstrucție','Blocuri 4 × 4, măști binare și reconstrucție VSR ×4.');
 const steps=[['Cadre HR','7 cadre RGB'],['Blocuri 4 × 4','16 pixeli / bloc'],['Măști dinamice','1 sau 2 măsurări / cadru'],['Cadre LR','1 sau 2 imagini ×4 mai mici'],['Rețea VSR','cadre HR estimate']];
 steps.forEach((a,i)=>{let x=0.81+i*2.55;rect(s,x,2.16,2.18,2.56,i%2?C.panel2:C.panel);txt(s,a[0],x+0.15,2.43,1.88,0.57,20,i===2?C.amber:C.cyan,{bold:true,align:'center'});txt(s,a[1],x+0.17,3.35,1.84,0.86,16,C.white,{align:'center'});if(i<4)arrow(s,x+2.23,3.21,0.30);});
 rect(s,1.02,5.44,11.26,0.74,C.panel2);txt(s,'Compresia se face la achiziție; măsurările rezultate rămân organizate ca imagini.',1.32,5.58,10.67,0.43,20,C.white,{align:'center'});
 notes(s,'Timp: 65 de secunde. Imaginea de înaltă rezoluție este împărțită în blocuri de patru pe patru pixeli. Senzorul face sume selective în fiecare bloc. Pentru fiecare cadru avem una sau două măsurări per bloc și obținem imagini cu dimensiunile spațiale de patru ori mai mici. Apoi aceste imagini joacă rolul intrării într-un sistem standard de video super-resolution. Avantajul este că datele comprimate se pot reprezenta tot ca imagini și pot fi prelucrate de infrastructura VSR.');
}
// 6 — one measurement
{
 let s=base('Varianta A: o măsurare pe cadru','Comparație la buget identic: 1 valoare / 16 pixeli / cadru → CR = 1/16.');
 txt(s,'16Sum: aceeași mască în fiecare cadru',0.92,1.88,5.2,0.45,21,C.amber,{bold:true});txt(s,'Random: 8 pixeli selectați, mască variabilă',6.70,1.88,5.6,0.45,21,C.cyan,{bold:true});
 mask(s,Array.from({length:4},()=>[1,1,1,1]),1.72,2.58,2.4,C.amber);
 mask(s,[[1,0,1,0],[1,1,0,0],[0,1,0,1],[1,0,0,1]],7.77,2.58,2.4);
 tag(s,'100% pixeli folosiți',1.31,5.25,3.25,C.amber);tag(s,'50% fill-factor',7.3,5.25,3.27,C.red);
 txt(s,'Avantaj: diversitate temporală. Cost: scade cantitatea de lumină folosită într-un cadru.',1.01,6.16,11.2,0.54,20,C.white,{align:'center'});
 notes(s,'Timp: 70 de secunde. În cazul unei singure măsurări, binningul standard 16Sum însumează toți cei 16 pixeli din bloc. O mască aleatoare folosește doar opt, iar configurația sa se schimbă între cadre. Ambele dau o valoare per bloc per cadru: raportul de măsurări este o șaisprezecime. Masca aleatoare furnizează diversitate pentru reconstrucție, dar la nivelul unui cadru renunță la jumătate din pixeli. Fill-factor redus înseamnă sensibilitate și SNR potențial mai slabe, deci autorii propun și soluția complementară.');
}
// 7 — two measurements
{
 let s=base('Varianta B: două măsurări complementare','Două sume diferite în același bloc: A + B acoperă toți cei 16 pixeli.');
 const a=[[1,1,0,0],[1,1,0,0],[1,1,0,0],[1,1,0,0]],b=a.map(row=>row.map(v=>1-v));
 mask(s,a,1.09,2.47,2.20,C.cyan);txt(s,'+',3.55,3.22,0.5,0.55,37,C.white,{bold:true,align:'center'});mask(s,b,4.36,2.47,2.20,C.amber);txt(s,'=',6.74,3.22,0.5,0.55,37,C.white,{bold:true,align:'center'});mask(s,Array.from({length:4},()=>[1,1,1,1]),7.53,2.47,2.20,C.green);
 txt(s,'masca A',1.34,4.88,1.8,0.4,16,C.cyan,{align:'center'});txt(s,'complementul B',4.57,4.88,1.9,0.4,16,C.amber,{align:'center'});txt(s,'toți pixelii',7.72,4.88,1.84,0.4,16,C.green,{align:'center'});
 card(s,10.16,2.43,2.29,2.85,'CR = 1/8','2 valori pentru fiecare bloc 4 × 4, per cadru.',C.green);
 txt(s,'Fill-factor 100% • suma A+B reproduce 16Sum • este nevoie de două căi de citire.',1.02,5.86,11.24,0.68,20,C.white,{align:'center'});
 notes(s,'Timp: 70 de secunde. Fiecare pixel participă la exact una dintre cele două măsurări. Prima sumă folosește masca A, a doua folosește complementul său B; împreună reproduc 16Sum. Astfel păstrăm tot semnalul colectat, adică fill-factor sută la sută, iar perechea conține mai multe informații pentru rețea decât suma finală unică. Avem două măsurări pentru 16 pixeli, deci raportul este o optime și fluxul de date se dublează față de varianta 1 pe 16. Implementarea hardware cere suplimentar elemente de citire și conversie.');
}
// 8 — designs
{
 let s=base('Ce măști se compară?', 'Două măsurări pe cadru; toate folosesc perechi complementare.');
 const data=[['8Sum fix','dreptunghiuri, fără variație',C.red],['8Sum alternant','coloane ↔ rânduri',C.amber],['Random','măști binare variabile',C.cyan],['Tetromino','forme T; varianta extinsă diversifică tiparele',C.green]];
 data.forEach((d,i)=>{let y=1.85+i*1.12;rect(s,0.91,y,11.52,0.90,i%2?C.panel2:C.panel);rect(s,1.02,y+0.13,0.09,0.65,d[2]);txt(s,d[0],1.32,y+0.17,3.4,0.50,22,d[2],{bold:true});txt(s,d[1],4.73,y+0.17,7.4,0.5,18,C.white);});
 txt(s,'Ipoteză: schimbarea tiparelor între cadre oferă măsurări mai informative.',1.16,6.57,11.02,0.4,19,C.white,{align:'center'});
 notes(s,'Timp: 60 de secunde. Configurația 8Sum fix corespunde unei împărțiri regulate; autorii o folosesc drept reper. Apoi alternează grupările pe coloane și rânduri. Măștile aleatoare oferă diversitate temporală. Tetromino folosește grupări în formă de T: sunt tipare cu pixeli adiacenți; în versiunea extinsă cresc posibilitățile între cadre. Testul compară nu doar neregularitatea spațială, ci și caracterul static ori dinamic al măștii.');
}
// 9 — reconstruction
{
 let s=base('Reconstrucția video', 'Se reutilizează rețele VSR; nu e nevoie de un solutor CS clasic.');
 card(s,0.84,1.85,5.50,3.59,'1 măsurare / cadru','Imaginea LR intră în BasicVSR, RVRT sau IART. Modelele sunt antrenate pentru fiecare tip de eșantionare.',C.cyan);
 card(s,6.60,1.85,5.85,3.59,'2 măsurări / cadru','Cele două imagini RGB se concatenează: intrare cu 6 canale. Se adaptează prima/ultima parte a BasicVSR; convoluția 1×1 produce RGB.',C.green);
 rect(s,1.22,5.73,10.88,0.75,C.panel2);txt(s,'Interpolarea biliniară inițială este suficientă conform ablațiilor raportate.',1.45,5.91,10.4,0.39,18,C.amber,{align:'center'});
 notes(s,'Timp: 75 de secunde. Pentru o măsurare pe cadru, se pot folosi direct arhitecturi consacrate precum BasicVSR, RVRT și IART, cu instruire adecvată datelor de intrare. Pentru două măsurări, două imagini RGB sunt concatenate pe canale: în loc de trei sunt șase canale. BasicVSR este adaptat la intrare și ieșire; o convoluție punctuală reduce ieșirea la RGB. Secvența nu devine de două ori mai lungă. Autorii arată în ablațiile din materialul suplimentar că interpolarea biliniară inițială este suficientă.');
}
// 10 — results one
{
 let s=base('Rezultate: o măsurare / cadru', 'PSNR (dB), VSR ×4, CR = 1/16; mai mare înseamnă mai bine.');
 const head=['Model','Bicubic','16Sum','Random','Câștig Random vs 16Sum'];const rows=[['BasicVSR','37,30','37,82','39,00','+1,18 dB'],['RVRT','38,10','39,07','40,65','+1,58 dB'],['IART','38,09','39,37','40,99','+1,62 dB']];
 let xx=[0.92,3.32,5.49,7.60,9.68],ww=[2.36,2.08,2.04,2.02,2.75];head.forEach((h,i)=>{rect(s,xx[i],2.04,ww[i],0.67,C.panel2);txt(s,h,xx[i]+0.07,2.16,ww[i]-0.14,0.38,i===4?13:17,C.cyan,{bold:true,align:i?'center':'left'});});
 rows.forEach((row,r)=>row.forEach((v,i)=>{rect(s,xx[i],2.78+r*0.91,ww[i],0.8,r%2?C.panel2:C.panel);txt(s,v,xx[i]+0.08,2.97+r*0.91,ww[i]-0.16,0.43,19,i===3||i===4?C.green:C.white,{bold:i===3||i===4,align:i?'center':'left'});}));
 txt(s,'Dataset: Vimeo-90K-T   •   toate cele trei rețele câștigă folosind măști Random.',1.03,5.88,11.19,0.54,20,C.white,{align:'center'});
 txt(s,'Sursă: Katz et al., Tabelul 1. Setarea Random utilizează numai 50% din pixeli per cadru.',1.10,6.51,10.95,0.34,13,C.muted,{align:'center'});
 notes(s,'Timp: 70 de secunde. Aceste numere sunt PSNR, în decibeli, pe setul Vimeo-90K-T. La același raport de măsurări de o șaisprezecime, Random este mai bun decât 16Sum cu aproximativ 1,18 până la 1,62 dB, în funcție de model. Rezultatul se repetă și pe Vid4 și în ceilalți indicatori raportați, SSIM și LPIPS. Important: beneficiul reconstrucției este observat într-o simulare comparativă, însă măștile Random folosesc doar jumătate din pixeli pe cadru. Pentru păstrarea sensibilității trecem la două măsurări complementare.');
}
// 11 — results two
{
 let s=base('Rezultate: două măsurări / cadru', 'BasicVSR, VSR ×4, CR = 1/8; date din Tabelul 2.');
 const names=['8Sum fix','8Sum alternant','Random dinamic','Tetromino basic','Tetromino extins'];const vals=[39.2199,40.7928,41.0379,41.4386,41.5675];
 names.forEach((n,i)=>{let y=1.96+i*0.79;txt(s,n,0.96,y+0.07,2.56,0.43,17,C.white);rect(s,3.66,y+0.12,(vals[i]-38.4)*2.4,0.42,i===4?C.green:i===0?C.red:C.cyan);txt(s,vals[i].toFixed(2).replace('.',',')+' dB',11.32,y+0.02,1.0,0.52,17,i===4?C.green:C.white,{bold:true,align:'right'});});
 rect(s,1.00,6.16,11.29,0.52,C.panel2);txt(s,'Tetromino extins vs 8Sum fix: +2,35 dB pe Vimeo-90K-T; pe Vid4: 31,86 vs 28,98 dB.',1.22,6.22,10.90,0.39,17,C.green,{bold:true,align:'center'});
 notes(s,'Timp: 75 de secunde. Aici comparăm numai variante cu două măsurări, deci același raport, o optime, și aceeași rețea BasicVSR. 8Sum fix obține 39,22 dB pe Vimeo-90K-T, alternarea simplă ajunge la 40,79, Random dinamic la 41,04, iar Tetromino extins la 41,57 dB: un câștig de 2,35 decibeli față de 8Sum fix. Pe Vid4 avem 31,86 versus 28,98 decibeli. Tabelul arată și că Tetromino static sau Random static sunt sub versiunile dinamice, semn că variația între cadre contează.');
}
// 12 — evaluation
{
 let s=base('Ce înseamnă rezultatele — și ce lipsește?', 'Interpretarea trebuie raportată la condițiile experimentale din articol.');
 card(s,0.83,1.80,5.65,3.99,'Puncte tari','• Câștiguri consecvente pe Vimeo-90K-T și Vid4.\n• Cadre LR de tip imagine, compatibile cu VSR.\n• Măștile complementare păstrează toți pixelii.',C.green);
 card(s,6.75,1.80,5.65,3.99,'Limite practice','• Senzor cu măști schimbate între cadre; nu e demonstrat un prototip.\n• Două măsurări necesită mai mult hardware și trafic decât una.\n• Reconstrucția depinde de asemănarea temporală și antrenare.',C.amber);
 txt(s,'F = 7 cadre în configurațiile evaluate • antrenare pe Vimeo-90K • testare pe Vimeo-90K-T și Vid4.',0.95,6.12,11.4,0.52,17,C.muted,{align:'center'});
 notes(s,'Timp: 70 de secunde. Rezultatele susțin ipoteza că modul de achiziție contează. Metoda are măsurări reprezentabile ca imagini și varianta complementară valorifică toți pixelii. Totuși, lucrarea evaluează experimental pe seturi de date video, nu raportează un senzor fizic complet demonstrat; măștile schimbate de la cadru la cadru cer adaptarea electronicii. Două măsurări per bloc înseamnă mai mult trafic decât una și componente de citire suplimentare. Ca orice metodă VSR, depinde de relația dintre cadre și de condițiile în care modelul a fost antrenat.');
}
// 13 — takeaways
{
 let s=base('Concluzie', 'Ideea de reținut în trei propoziții.');
 card(s,0.86,1.86,11.61,1.27,'01  Proiectează măsurarea','Calitatea reconstrucției depinde și de ce informație păstrează senzorul.',C.cyan);
 card(s,0.86,3.35,11.61,1.27,'02  Folosește timpul','Măștile dinamice oferă informație complementară între cadre.',C.green);
 card(s,0.86,4.85,11.61,1.27,'03  Respectă constrângerile','Măștile complementare păstrează sensibilitatea, dar dublează măsurările.',C.amber);
 txt(s,'Referință: Katz et al., „Rethinking Compressive Sensing: A Compression Framework for Video Super-Resolution”, CVPRW 2025.',0.98,6.54,11.28,0.47,13,C.muted,{align:'center'});
 notes(s,'Timp: 45 de secunde. Concluzia este că nu trebuie tratată achiziția ca un pas fix și reconstrucția ca singura problemă de optimizat. Eșantionarea dinamică poate păstra informație complementară de la un cadru la altul, inclusiv acolo unde mișcarea scenei nu ajută mult. Varianta cu două măști complementare elimină pierderea sensibilității asociată măștilor parțiale, în schimbul unui volum de măsurări mai mare și al unor cerințe suplimentare pentru senzor. Mulțumesc.');
}

const out=path.join(__dirname,'Prezentare_Katz_CS_VSR_2025_RO.pptx');
pptx.writeFile({fileName:out});
