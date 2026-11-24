"""
Fabrika mapa za Moju prognozu.
Preuzima model ICON-EU (DWD otvoreni podaci, ~7 km) i crta glatke slike
temperature i padavina (kiša/sneg) za Balkan, sat po sat, 48 h unapred.
Izlaz: izlaz/t_XXX.webp, izlaz/p_XXX.webp, izlaz/index.json
"""
import bz2, datetime as dt, io, json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import ndimage
from PIL import Image
import eccodes

BAZA = "https://opendata.dwd.de/weather/nwp/icon-eu/grib"
SATI = int(os.environ.get("SATI", "48"))
# oblast (stepeni): sever, jug, zapad, istok
S, J, Z, I = 50.0, 38.0, 12.0, 30.0
SIRINA = 1000                      # px slike
IZLAZ = os.environ.get("IZLAZ", "izlaz")

GRADOVI = [
    ["Novi Sad",45.267,19.833],["Beograd",44.787,20.457],["Zrenjanin",45.381,20.391],["Subotica",46.100,19.667],
    ["Sombor",45.774,19.112],["Kikinda",45.829,20.465],["Pančevo",44.871,20.640],["Vršac",45.117,21.300],
    ["Šabac",44.756,19.694],["Valjevo",44.270,19.885],["Kragujevac",44.012,20.912],["Čačak",43.891,20.350],
    ["Kraljevo",43.724,20.687],["Užice",43.856,19.842],["Niš",43.320,21.896],["Leskovac",42.998,21.946],
    ["Novi Pazar",43.137,20.512],["Zaječar",43.904,22.285],["Vranje",42.551,21.900],["Zlatibor",43.729,19.700],
    ["Kopaonik",43.286,20.810],["Segedin",46.253,20.148],["Temišvar",45.748,21.208],["Arad",46.186,21.312],
    ["Rešica",45.300,21.889],["Osijek",45.554,18.696],["Vukovar",45.351,19.002],["Tuzla",44.538,18.668],
    ["Bijeljina",44.757,19.214],["Sarajevo",43.856,18.413],["Banja Luka",44.772,17.191],["Mostar",43.343,17.808],
    ["Zagreb",45.815,15.982],["Split",43.508,16.440],["Dubrovnik",42.650,18.094],["Rijeka",45.327,14.442],
    ["Ljubljana",46.056,14.506],["Budimpešta",47.498,19.040],["Debrecin",47.531,21.627],["Pečuj",46.073,18.233],
    ["Kečkemet",46.896,19.689],["Krajova",44.330,23.795],["Bukurešt",44.427,26.103],["Kluž",46.770,23.590],
    ["Vidin",43.990,22.873],["Sofija",42.698,23.322],["Plovdiv",42.136,24.745],["Varna",43.214,27.915],
    ["Skoplje",41.998,21.425],["Ohrid",41.117,20.802],["Podgorica",42.441,19.263],["Budva",42.288,18.842],
    ["Priština",42.663,21.164],["Tirana",41.327,19.819],["Solun",40.640,22.944],["Beč",48.208,16.373],
    ["Bratislava",48.149,17.107],["Venecija",45.441,12.316],["Bari",41.117,16.872],
]

# ---------- palete (stil kao Vreme & Radar) ----------
# temperatura: puna polja, nova boja na svaki stepen
T_PAL = [(-24,(110,30,150)),(-18,(90,50,190)),(-12,(60,80,215)),(-8,(45,115,230)),(-5,(55,150,240)),
         (-2,(90,190,245)),(0,(150,225,245)),(1,(60,170,70)),(4,(85,190,45)),(7,(125,208,30)),(10,(185,220,20)),
         (13,(240,220,25)),(16,(250,185,30)),(19,(245,150,30)),(22,(240,115,30)),(26,(225,70,35)),(30,(200,35,40)),
         (35,(150,15,60)),(40,(110,5,80))]
# kiša (mm/h): plave stepenice, jaka kiša ljubičasto
K_STEP = [(0.1,(175,235,255)),(0.3,(120,210,250)),(0.7,(60,170,245)),(1.5,(30,120,230)),(3,(20,80,210)),
          (6,(15,45,170)),(12,(130,45,205)),(25,(215,45,200))]
# sneg: belo do roze
SN_STEP = [(0.1,(245,245,255)),(0.4,(255,220,245)),(1.2,(250,180,235)),(3,(230,130,220))]

def oboji(v, pal):
    x = np.array([p[0] for p in pal], float)
    c = np.array([p[1] for p in pal], float)
    return np.stack([np.interp(v, x, c[:, k]) for k in range(3)], -1)

def stepenice(v, st):
    """Ravne boje po stepenicama (kao radar)."""
    out = np.zeros(v.shape + (3,))
    for g, c in st:
        out[v >= g] = c
    return out

# ---------- preuzimanje ----------
def url(run, var, h):
    d = run.strftime("%Y%m%d%H")
    return f"{BAZA}/{run:%H}/{var}/icon-eu_europe_regular-lat-lon_single-level_{d}_{h:03d}_{var.upper()}.grib2.bz2"

def preuzmi(u, pokusaja=3):
    for p in range(pokusaja):
        try:
            with urllib.request.urlopen(u, timeout=60) as r:
                return bz2.decompress(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
        except Exception:
            pass
    return None

def dekodiraj(b):
    g = eccodes.codes_new_from_message(b)
    try:
        ni = eccodes.codes_get(g, "Ni"); nj = eccodes.codes_get(g, "Nj")
        la1 = eccodes.codes_get(g, "latitudeOfFirstGridPointInDegrees")
        la2 = eccodes.codes_get(g, "latitudeOfLastGridPointInDegrees")
        lo1 = eccodes.codes_get(g, "longitudeOfFirstGridPointInDegrees")
        lo2 = eccodes.codes_get(g, "longitudeOfLastGridPointInDegrees")
        v = eccodes.codes_get_values(g).reshape(nj, ni)
    finally:
        eccodes.codes_release(g)
    if lo1 > 180: lo1 -= 360
    if lo2 > 180: lo2 -= 360
    if la1 > la2:                      # neka ide od juga ka severu
        v = v[::-1]; la1, la2 = la2, la1
    return v, (la1, la2, lo1, lo2)

def poslednji_run():
    sad = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    for nazad in range(0, 30, 6):
        t = sad - dt.timedelta(hours=4 + nazad)
        run = t.replace(hour=t.hour - t.hour % 6, minute=0, second=0, microsecond=0)
        if preuzmi(url(run, "t_2m", SATI)) is not None:
            return run
    sys.exit("Nema dostupnog ICON-EU proračuna")

# ---------- crtanje ----------
def merkator(lat): return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
VISINA = int(round(SIRINA * (merkator(S) - merkator(J)) / np.radians(I - Z)))

def mreza_izlaza(granice, nj, ni):
    la1, la2, lo1, lo2 = granice
    dla = (la2 - la1) / (nj - 1); dlo = (lo2 - lo1) / (ni - 1)
    y = np.linspace(merkator(S), merkator(J), VISINA)              # odozgo nadole
    lat = np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)
    lon = np.linspace(Z, I, SIRINA)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    return np.array([(LA - la1) / dla, (LO - lo1) / dlo])

def uzorkuj(v, koord, red=3):
    return ndimage.map_coordinates(v, koord, order=red, mode="nearest")

def slika(rgb, alfa, ime):
    a = np.dstack([np.clip(rgb, 0, 255), np.clip(alfa, 0, 255)]).astype(np.uint8)
    Image.fromarray(a, "RGBA").save(os.path.join(IZLAZ, ime), "WEBP", quality=82, method=6)

def main():
    os.makedirs(IZLAZ, exist_ok=True)
    run = poslednji_run()
    print("Proračun:", run)
    koraci = list(range(0, SATI + 1))
    poslovi = [(var, h) for h in koraci for var in ("t_2m", "tot_prec", "snow_gsp", "snow_con", "clct")]
    with ThreadPoolExecutor(12) as ex:
        sirovo = dict(zip(poslovi, ex.map(lambda p: preuzmi(url(run, *p)), poslovi)))
    koord = None
    vremena = []
    gradT, gradN, gradP, gradS = ([[] for _ in GRADOVI] for _ in range(4))
    preth = None
    for h in koraci:
        tb = sirovo[("t_2m", h)]
        if tb is None: break
        t, gr = dekodiraj(tb)
        t = t - 273.15
        if koord is None:
            koord = mreza_izlaza(gr, *t.shape)
            la1, la2, lo1, lo2 = gr
            dla = (la2 - la1) / (t.shape[0] - 1); dlo = (lo2 - lo1) / (t.shape[1] - 1)
            gk = np.array([[(g[1] - la1) / dla for g in GRADOVI], [(g[2] - lo1) / dlo for g in GRADOVI]])
        # temperatura
        tt = uzorkuj(t, koord)
        # puna polja po 1 °C, ivice blago omekšane
        tb = np.floor(tt) + 0.5
        boja_t = ndimage.uniform_filter(oboji(tb, T_PAL), size=(3, 3, 1))
        slika(boja_t, np.full(tt.shape, 235.0), f"t_{h:03d}.webp")
        # oblaci: beli, providni
        cb = sirovo[("clct", h)]
        if cb is not None:
            cl = ndimage.gaussian_filter(dekodiraj(cb)[0], 0.7)
            cc = uzorkuj(cl, koord)
            slika(np.full(cc.shape + (3,), 238.0), np.clip((cc - 20) / 80, 0, 1) ** 1.3 * 140, f"c_{h:03d}.webp")
            for k, v in enumerate(ndimage.map_coordinates(cl, gk, order=1)):
                gradN[k].append(int(round(float(np.clip(v, 0, 100)))))
        else:
            for k in range(len(GRADOVI)): gradN[k].append(None)
        for k, v in enumerate(ndimage.map_coordinates(t, gk, order=1)):
            gradT[k].append(round(float(v), 1))
        # padavine: razlika akumulacije u odnosu na prethodni sat
        def ak(var):
            b = sirovo[(var, h)]
            return dekodiraj(b)[0] if b is not None else None
        sada = {v: ak(v) for v in ("tot_prec", "snow_gsp", "snow_con")}
        if preth is not None and sada["tot_prec"] is not None:
            pr = np.maximum(0, sada["tot_prec"] - preth["tot_prec"])
            sn = np.zeros_like(pr)
            for v in ("snow_gsp", "snow_con"):
                if sada[v] is not None and preth[v] is not None:
                    sn += np.maximum(0, sada[v] - preth[v])
            pr = ndimage.gaussian_filter(pr, 1.0)
            udeo = ndimage.gaussian_filter(np.where(pr > 0.02, sn / np.maximum(pr, 1e-6), 0), 1.0)
            for k, (pv, uv) in enumerate(zip(ndimage.map_coordinates(pr, gk, order=1), ndimage.map_coordinates(udeo, gk, order=1))):
                gradP[k].append(round(float(pv), 1)); gradS[k].append(bool(uv > 0.5))
            pp = np.maximum(0, uzorkuj(pr, koord))
            ud = uzorkuj(udeo, koord, 1)
            boja = np.where((ud > 0.5)[..., None], stepenice(pp, SN_STEP), stepenice(pp, K_STEP))
            boja = ndimage.uniform_filter(boja, size=(3, 3, 1))
            alfa = ndimage.uniform_filter(np.where(pp >= 0.1, 235.0, 0.0), size=3)
            slika(boja, alfa, f"p_{h:03d}.webp")
        else:
            for k in range(len(GRADOVI)): gradP[k].append(0.0); gradS[k].append(False)
        if sada["tot_prec"] is not None:
            preth = sada
        vremena.append((run + dt.timedelta(hours=h)).strftime("%Y-%m-%dT%H:00Z"))
    json.dump({
        "proracun": run.strftime("%Y-%m-%dT%H:00Z"),
        "napravljeno": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%dT%H:%MZ"),
        "granice": {"sever": S, "jug": J, "zapad": Z, "istok": I},
        "vremena": vremena,             # t_000 = vremena[0]; p_ postoji od 1. sata
        "gradovi": [{"ime": g[0], "lat": g[1], "lon": g[2], "t": gradT[k], "n": gradN[k], "p": gradP[k], "s": gradS[k]} for k, g in enumerate(GRADOVI)],
    }, open(os.path.join(IZLAZ, "index.json"), "w"), ensure_ascii=False)
    print("Gotovo:", len(vremena), "sati")

if __name__ == "__main__":
    main()
