
L = 0.0008;
W = 0.0004;
h = 0.000012;

Point(1)  = {-L,   -W/2, 0, h};
Point(2)  = {-W/2, -W/2, 0, h};
Point(3)  = {-W/2, -L,   0, h};
Point(4)  = { W/2, -L,   0, h};
Point(5)  = { W/2, -W/2, 0, h};
Point(6)  = { L,   -W/2, 0, h};
Point(7)  = { L,    W/2, 0, h};
Point(8)  = { W/2,  W/2, 0, h};
Point(9)  = { W/2,  L,   0, h};
Point(10) = {-W/2,  L,   0, h};
Point(11) = {-W/2,  W/2, 0, h};
Point(12) = {-L,    W/2, 0, h};

Line(1)  = {1,2};
Line(2)  = {2,3};
Line(3)  = {3,4};
Line(4)  = {4,5};
Line(5)  = {5,6};
Line(6)  = {6,7};
Line(7)  = {7,8};
Line(8)  = {8,9};
Line(9)  = {9,10};
Line(10) = {10,11};
Line(11) = {11,12};
Line(12) = {12,1};

Curve Loop(100) = {1,2,3,4,5,6,7,8,9,10,11,12};
Plane Surface(400) = {100};

Physical Surface("fluid") = {400};
Physical Line("inlet1") = {12};
Physical Line("inlet2") = {6};
Physical Line("outlet1") = {3};
Physical Line("outlet2") = {9};
Physical Line("wall") = {1,2,4,5,7,8,10,11};

Mesh.MeshSizeMin = h;
Mesh.MeshSizeMax = h;
Mesh.Algorithm = 6;
Mesh.Smoothing = 30;
