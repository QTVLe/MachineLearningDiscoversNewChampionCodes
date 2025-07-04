SetNthreads(72);

prime := 8; // this is a prime
primitive := PrimitiveElement(FiniteField(prime)); //gives a primitive root
allvertices := [<x,y> : x,y in [0..prime-2]]; //gives an ordered set of all vertices in the box

generalisedToricMatrix := function(vertices);
    M := KMatrixSpace(FiniteField(prime), #vertices, (prime-1)^2);
    rows := [primitive^((allvertices[j][1])*vertices[i][1] + (allvertices[j][2])*vertices[i][2]): j in [1..(prime-1)^2], i in [1..#vertices]];
    toricmatrix := M ! rows;
    return toricmatrix;
end function;         
                               
sortvertices1 := function(v,w);
            if (v[1] lt w[1]) then 
                        return -1; 
            elif (v[1] gt w[1]) then
                        return 1;
            elif (v[2] lt w[2]) then
                        return -1;
            elif (v[2] gt w[2]) then
                        return 1;
            else
                        return 0;
            end if;
end function;

vertex_order1 := function(vertices);
            sorted_coordinates := Sort(vertices, func<v, w | sortvertices1(v,w) >);
            return sorted_coordinates;
end function;                                            
                                                                                                                  
generateRandomVertex := function();
    i := Random(0,prime-1);
    j := Random(0,prime-1);
    vertex := <i,j>;
    return vertex;
end function;

generateRandomVertices := function();
    number := Random(3,(prime-1)^2);
    vertices := SetToSequence({generateRandomVertex() : i in [1..number]});
    verticesordered1 := vertex_order1(vertices);
    return verticesordered1;
end function;

generateSetsofVertices := function(number);
            set := {};
            setofallvertices := SequenceToSet(allvertices);
            while #set lt 20000 do //replace number to generate a dataset of that size
                  vertices := RandomSubset(setofallvertices, number);
                  vertices := vertex_order1(SetToSequence(vertices));
                  Include(~set, vertices);
                  #set;
            end while;
            verticesset := SetToSequence(set);
            return verticesset;
end function;
            
minimumdistance := function(code);
         SetVerbose("Code",true); //this gives information on the progress of the computation, turn false to stop this
         SetSeed(1);
         mindistance := MinimumWeight(code: MaximumTime := 60);
         return mindistance;
end function;

generateToricCode := function(vertices);
    toricmatrix := generalisedToricMatrix(vertices); //makes a matrix of vertices
    toriccode := LinearCode(toricmatrix); //makes toric code from this matrix
    generator_matrix := GeneratorMatrix(toriccode); // gives the current vector space basis of code
    generator := Generators(toriccode);
    mindistance := minimumdistance(toriccode); //calculates minimum distance of toric code
    print(mindistance);
    list := [*mindistance, generator*];
    stringElements := [Sprint(e) : e in list];
    joinedElements := &cat[ stringElements[i] cat (i lt #stringElements select "; " else "") : i in [1..#stringElements] ];
    return [joinedElements];
end function;

for number in [40,41,42,43,44,45] do //change numbers in the list to cover a certain ranges
            verticesset := generateSetsofVertices(number);
            dataset := [];                                                                                                                           
            for vertices in verticesset do
                        Append(~dataset,generateToricCode(vertices));
            end for;
            Write("datasetnew_" cat IntegerToString(number) cat "_F8.txt", dataset);
end for;
