
doc = app.documents;

// // select destination folder
var destFolder = Folder.selectDialog ('Save Image', Folder.myDocuments);

// fileOut = new File("D:/Log.txt")
// fileOut.lineFeed    = "windows"
// fileOut.open("w", "TEXT", "????")

// nameFile = 
// var left_top        = 20;
// var left_bottom     = 30;
// var right_top       = 40;
// var right_bottom    = 50;

// write out a header
// fileOut.write("name_file" + "\t" + "x_top" + "\t" + "x_bottom" + "\t" + "y_left" + "\t" + "y_right" + "\t" + "px_center" + "\t" + "x_rate" + "\t" + "y_rate"  + "\n")
// fileOut.write("name_file" + "\t" + "px_center" + "\t" + "x_rate" + "\t" + "y_rate"  + "\n")

for(var i = 0; i < doc.length; i++){
    app.activeDocument = doc[i];

    // fill sky color selection white
    app.activeDocument.selection.fill(app.foregroundColor)

    // clear selection
    // app.activeDocument.selection.clear();

    // invert selection
    app.activeDocument.selection.invert();

    // fill land color selection black
    app.activeDocument.selection.fill(app.backgroundColor)

    // deselect selection
    app.activeDocument.selection.deselect();

    // save mask

    // JPG Options;
    jpgSaveOptions = new JPEGSaveOptions();  
    jpgSaveOptions.embedColorProfile = true;  
    jpgSaveOptions.formatOptions = FormatOptions.STANDARDBASELINE;  
    jpgSaveOptions.matte = MatteType.NONE;  
    jpgSaveOptions.quality = 8;   

    var theName = doc[i].name.match(/(.*)\.[^\.]+$/)[1];

    var outputFile = new File(destFolder + '/' + theName + '.jpg')
    doc[i].saveAs(outputFile, jpgSaveOptions, true, Extension.LOWERCASE);
    
}

// fileOut.close()


function quickSel (x, y, tol){
    var idsetd = charIDToTypeID( "setd" );
        var desc2 = new ActionDescriptor();
        var idnull = charIDToTypeID( "null" );
            var ref1 = new ActionReference();
            var idChnl = charIDToTypeID( "Chnl" );
            var idfsel = charIDToTypeID( "fsel" );
            ref1.putProperty( idChnl, idfsel );
        desc2.putReference( idnull, ref1 );
        var idT = charIDToTypeID( "T   " );
            var desc3 = new ActionDescriptor();
            var idHrzn = charIDToTypeID( "Hrzn" );
            var idPxl = charIDToTypeID( "#Pxl" );
            desc3.putUnitDouble( idHrzn, idPxl, x );
            var idVrtc = charIDToTypeID( "Vrtc" );
            var idPxl = charIDToTypeID( "#Pxl" );
            desc3.putUnitDouble( idVrtc, idPxl, y);
        var idPnt = charIDToTypeID( "Pnt " );
        // alert('desc3: ' + desc3)
        desc2.putObject( idT, idPnt, desc3 );
        var idTlrn = charIDToTypeID( "Tlrn" );
        desc2.putInteger( idTlrn, tol);
        var idAntA = charIDToTypeID( "AntA" );
        desc2.putBoolean( idAntA, true );
        var idCntg = charIDToTypeID( "Cntg" );
        desc2.putBoolean( idCntg, true );
    executeAction( idsetd, desc2, DialogModes.NO );
    };


// Add selection
function addSel(x, y, tol){
        var idAddT = charIDToTypeID( "AddT" );
        var desc516 = new ActionDescriptor();
        var idnull = charIDToTypeID( "null" );
            var ref68 = new ActionReference();
            var idChnl = charIDToTypeID( "Chnl" );
            var idfsel = charIDToTypeID( "fsel" );
            ref68.putProperty( idChnl, idfsel );
        desc516.putReference( idnull, ref68 );
        var idT = charIDToTypeID( "T   " );
            var desc517 = new ActionDescriptor();
            var idHrzn = charIDToTypeID( "Hrzn" );
            var idPxl = charIDToTypeID( "#Pxl" );
            desc517.putUnitDouble( idHrzn, idPxl, x );
            var idVrtc = charIDToTypeID( "Vrtc" );
            var idPxl = charIDToTypeID( "#Pxl" );
            desc517.putUnitDouble( idVrtc, idPxl, y );
        var idPnt = charIDToTypeID( "Pnt " );
        desc516.putObject( idT, idPnt, desc517 );
        var idTlrn = charIDToTypeID( "Tlrn" );
        desc516.putInteger( idTlrn, tol );
        var idAntA = charIDToTypeID( "AntA" );
        desc516.putBoolean( idAntA, true );
    executeAction( idAddT, desc516, DialogModes.NO );

}


// get color at specific pixel

function getColor(x,y,doc){
    #target photoshop

    // Add a Color Sampler at a given x and y coordinate in the image.
    app.activeDocument = doc
    var pointSample = app.activeDocument.colorSamplers.add([(x - 1),(y - 1)]);

    // Obtain array of RGB values.
    var rgb = 0;
    rgb += Math.round(pointSample.color.rgb.red)
    rgb += Math.round(pointSample.color.rgb.green)
    rgb += Math.round(pointSample.color.rgb.blue)


    // Remove the Color Sampler.
    pointSample.remove();

    // Display the complete RGB values and each component color.
    // alert('RGB: ' + rgb)
    // alert('red: ' + rgb[0])
    // alert('green: ' + rgb[1])
    // alert('blue: ' + rgb[2])
    return rgb

}

function MinMaxScaler(arr){
    var average = 0
    var min_val = 1
    var max_val = 0 

    var min_index
    var max_index
    for(var x=0; x<arr.length; x++){
        // find average
        average += arr[x]

        // find min value
        if(arr[x] < min_val){
            min_val = arr[x]
            min_index = x
        }

        // find max value
        if(arr[x] > max_val){
            max_val = arr[x]
            max_index = x + 1
        }

    }
    // alert('arr: ' + arr + "\t" + 'max_val: ' + max_val)


    average = average/ arr.length

    var min_delta_val = 1
    var min_delta_index
    for(var x=0; x<arr.length; x++){
        var delta = Math.abs(arr[x] - average)

        if(delta < min_delta_val){
            min_delta_val = delta
            min_delta_index = x + 1
        }
    }

    var index_scaled = 0

    if(max_index <= arr.length/2){
        index_scaled = max_index/arr.length + (max_val - average)*1
    }else{
        index_scaled = max_index/arr.length - (max_val - average)*1

    }

    // var index_scaled = min_delta_index/arr.length + arr[min_delta_index] - average
    // alert(index_scaled)

    return index_scaled
}