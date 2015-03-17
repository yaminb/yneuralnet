A very easy to use and flexible Neural net class written in c#.  You specify your own data (from file, network, ...)
Everything from the activation funtion on is customizable.

I just couldn't find a neural net class I liked.  This one is nice and easy and simple to use :)  It's how I code.

The sample projects (YAITest and YAICompression) are probably the easiest to see how it works.  Basically, you just implement an interface that provides the neural net with its needed functions and it handles all the dirty work.

I don't know if anyone is going to use this, but I'm toying neural nets right now and this is a nice repo.

My first toy was YAICompression.  Basically, it tries to predict that next byte based on the input (previous bytes).  It does this bitwise, but I wasn't able to achieve any great compression beyond what zip or LZ compression could do.  But at least I know it works decently.

Maybe someone more mathematically inclined can figure out the proper number of hidden layers and nodes and all that mystery.


