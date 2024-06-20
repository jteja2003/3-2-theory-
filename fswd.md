# FSWD
# UNIT-1
## SVG AND CANVAS 
SVG (Scalable Vector Graphics) and Canvas are both HTML technologies used for drawing graphics, but they are fundamentally different in terms of how they work and what they are best suited for. Here’s a detailed comparison along with examples for both:
### SVG (Scalable Vector Graphics)
- **Description**: SVG is an XML-based format for describing two-dimensional vector graphics. Each element and attribute in the SVG can be animated, styled with CSS, and manipulated through JavaScript.
- **Best for**: Static images, complex illustrations, graphics that need to be interactive, and images that need to be scalable without losing quality.

#### Example of SVG:
```html
<!DOCTYPE html>
<html>
<head>
    <title>SVG Example</title>
</head>
<body>
    <h1>SVG Example</h1>
    <svg width="400" height="110">
        <rect width="300" height="100" style="fill:rgb(0,0,255);stroke-width:1;stroke:rgb(0,0,0)" />
        <circle cx="150" cy="50" r="40" stroke="green" stroke-width="4" fill="yellow" />
        <text x="150" y="105" font-family="Verdana" font-size="35" fill="red" text-anchor="middle">SVG</text>
    </svg>
</body>
</html>
```

### Canvas
- **Description**: Canvas is an HTML element used to draw graphics via JavaScript. It is resolution-dependent and supports rendering of both 2D and 3D graphics.
- **Best for**: Real-time graphics, animations, games, and when you need to frequently update the graphics.

#### Example of Canvas:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Canvas Example</title>
</head>
<body>
    <h1>Canvas Example</h1>
    <canvas id="myCanvas" width="400" height="110" style="border:1px solid #000000;"></canvas>
    <script>
        var canvas = document.getElementById('myCanvas');
        var ctx = canvas.getContext('2d');

        // Draw a blue rectangle
        ctx.fillStyle = 'rgb(0, 0, 255)';
        ctx.fillRect(10, 10, 300, 100);

        // Draw a yellow circle with a green border
        ctx.beginPath();
        ctx.arc(150, 60, 40, 0, 2 * Math.PI, false);
        ctx.fillStyle = 'yellow';
        ctx.fill();
        ctx.lineWidth = 4;
        ctx.strokeStyle = 'green';
        ctx.stroke();

        // Draw text
        ctx.font = '35px Verdana';
        ctx.fillStyle = 'red';
        ctx.textAlign = 'center';
        ctx.fillText('Canvas', 150, 105);
    </script>
</body>
</html>
```

### Key Differences:
1. **Rendering**:
   - SVG uses a retained-mode graphics model. This means it keeps track of the objects and can manipulate them independently.
   - Canvas uses an immediate-mode graphics model. This means it does not keep track of the objects and only renders the pixels.

2. **Interactivity and Animation**:
   - SVG elements can be styled with CSS and can respond to DOM events like clicks and mouse movements.
   - Canvas requires JavaScript for interactivity and animation, often involving manual management of drawing and redrawing elements.

3. **Scalability**:
   - SVG graphics are vector-based, which means they scale well without losing quality.
   - Canvas graphics are resolution-dependent and may not scale as well, becoming pixelated when enlarged.

4. **Performance**:
   - SVG is better suited for applications with fewer objects or complex static graphics.
   - Canvas performs better with a large number of objects and real-time rendering scenarios, such as games or simulations.

Choosing between SVG and Canvas depends on the specific requirements of your project. For static, high-quality graphics that require interaction, SVG is usually the better choice. For dynamic, high-performance graphics and real-time rendering, Canvas is generally more suitable.
