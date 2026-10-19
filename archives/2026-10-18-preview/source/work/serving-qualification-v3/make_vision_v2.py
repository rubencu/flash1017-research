"""Create a fresh geometric test image with pixel-exact square dimensions."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import zlib

HERE=Path(__file__).resolve().parent


def main():
    image=HERE/"fixtures/vision-shapes-v2.png"
    metadata=HERE/"vision-fixture-v2.json"
    revised=HERE/"validation-plan-02.json"
    if any(p.exists() for p in (image,metadata,revised)):
        raise FileExistsError("Preserve existing artifacts")
    w,h=512,384
    green=(20,145,65);blue=(30,80,220);red=(220,35,45)
    rectangles=[(60,170,124,234),(170,170,234,234),(280,170,344,234)]
    assert all(x1-x0 == y1-y0 == 64 for x0,y0,x1,y1 in rectangles)
    rows=[]
    for y in range(h):
        row=bytearray()
        for x in range(w):
            rgb=(255,255,255)
            if any((x-cx)**2+(y-85)**2<=38**2 for cx in (120,265)):rgb=blue
            if any(x0<=x<=x1 and y0<=y<=y1 for x0,y0,x1,y1 in rectangles):rgb=green
            if 170<=y<=245 and abs(x-425)<=(y-170)*0.6:rgb=red
            if 330<=y<=350 and 20<=x<=490:rgb=(225,225,225)
            row.extend(rgb)
        rows.append(row)
    # Check connected-component dimensions of the output pixels independently
    # of the rectangle construction before serializing the image.
    points={(x,y) for y,row in enumerate(rows) for x in range(w) if tuple(row[3*x:3*x+3])==green}
    bounds=[]
    while points:
        seed=points.pop();todo=[seed];component=[seed]
        while todo:
            x,y=todo.pop()
            for nxt in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if nxt in points:points.remove(nxt);todo.append(nxt);component.append(nxt)
        xs,ys=zip(*component);bounds.append([max(xs)-min(xs)+1,max(ys)-min(ys)+1,len(component)])
    assert sorted(bounds)==[[65,65,4225]]*3
    def chunk(kind,value):
        return struct.pack(">I",len(value))+kind+value+struct.pack(">I",zlib.crc32(kind+value)&0xffffffff)
    data=b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",w,h,8,2,0,0,0))
    data+=chunk(b"IDAT",zlib.compress(b"".join(b"\0"+bytes(r) for r in rows),9))+chunk(b"IEND",b"")
    image.write_bytes(data)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    record={"status":"frozen_unrun","path":"fixtures/vision-shapes-v2.png","sha256":digest(image),
            "width":w,"height":h,"expected":{"blue_circles":2,"green_squares":3,"red_triangles":1},
            "verified_green_components":sorted(bounds),"license":"Original diagnostic fixture, CC0-1.0",
            "created_at_utc":datetime.now(timezone.utc).isoformat(),"builder_sha256":digest(Path(__file__)),
            "supersedes":{"path":"fixtures/vision-shapes.png","sha256":digest(HERE/"fixtures/vision-shapes.png"),
                          "reason":"Visual QA found original green shapes were65x76, not squares. Original preserved; no model has consumed either image during preparation. Token fixtures and bench plans are unchanged."}}
    metadata.write_text(json.dumps(record,indent=2)+"\n")
    plan=json.loads((HERE/"validation-plan.json").read_text())
    plan["supersedes"]={"path":"validation-plan.json","sha256":digest(HERE/"validation-plan.json"),"scope":"Only correct the unrun vision image fixture; all token/bench plans and other checks unchanged."}
    for case in plan["http_cases"]:
        if case.get("image")=="fixtures/vision-shapes.png":case["image"]="fixtures/vision-shapes-v2.png"
    plan["generated_artifact_hashes"]["vision-fixture-v2.json"]=digest(metadata)
    revised.write_text(json.dumps(plan,indent=2)+"\n")
    print(json.dumps({"image_sha256":digest(image),"metadata_sha256":digest(metadata),"validation_plan_sha256":digest(revised),"pixel_square_checks":"passed"}))


if __name__=="__main__":main()
