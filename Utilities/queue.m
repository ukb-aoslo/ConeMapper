%Code (c) 2020 G. de Sercey
%Licensed under Creative Commons Attribution Share Alike 3.0 license, per the terms of Matlab Answers
%https://uk.mathworks.com/matlabcentral/termsofuse.html#answer
%
%Usage example:
%   q = queue;   %creates a queue with default initial capacity (1000 elements. grows as needed).
%   q.enqueue(struct('id', 1, 'value, 'aaa'));
%   q.enqueue([2 3 4 5 6]);
%   while q.Depth > 0
%       disp('element dequeued:');
%       disp(q.dequeue);
%   end
%
%Properties of the queue class:
%   - Depth: number of elements currently stored in the queue
%
%Methods of the class:
%   - qobj.enqueue(element): store one element in the queue
%   - element = qobj.dequeue: remove one element from the queue
%
%Static methods of the class:
%   - qarr = queue.MakeVectorOfQueues(sz): construct an array of distinct queues.
%Note that queue is a handle class.
classdef queue < handle
    properties (Dependent, SetAccess = private)
        Depth;  %Number of elements currently stored in the queue
    end
    properties (Access = private)
        Storage;  %cell array to store the queue
        Bounds;   %Bounds(1) is the index of the start of the queue, Bounds(2) is the index one past the end of the queue (insertion point). Indices are 0-based.
    end
    methods
        function this = queue(initialcapacity)
            %QUEUE Creates a queue object. The queue is heterogeneous and can store any type of data.
            %   q = queue;
            %   q = queue(initialcapacity);
            %initialcapacity: specifies how many objects the queue can stored initially. Default: 1000.
            %The queue double in size every time its capacity is reached.
            %The queue has one property, Depth, which indicates how many elements it currently stores.
%             arguments
%                 initialcapacity (1,1) {mustBeInteger, mustBePositive} = 1000
%             end
            this.Storage = cell(1, initialcapacity);
            this.Bounds = [0, 0];  
            %this.Bounds is used to track the ends of the queue. 
            %this.Bounds(2) can be greater than the size of this.Storage when the queue 'wraps' around to the start of storage.
            %Note these indices are 0-based as it simplifies the modulo operation to calculate true indices once Bounds(2) has 'wrapped' around.
            %It is guaranteed that this.Bounds(2) >= this.Bounds(1) since when this.Bounds(1) == this.Bounds(2), the queue is empty.
        end
        function depth = get.Depth(this)
            depth = diff(this.Bounds);
        end
        function enqueue(this, in)
            %ENQUEUE Adds an element to the end of the queue.
            %   q.enqueue(in)
            %q: scalar queue object.
            %in: element to store in the queue.  This increases the depth of the queue by one.
            assert(numel(this) == 1, 'queue:enqueue:nonscalarqueue', 'can only enqueue in a scalar queue.');
            capacity = numel(this.Storage);
            if this.Depth == capacity
                %no more room in the queue. Double its size. May need some copying if the queue has wrapped around.
                this.Storage = [this.Storage, cell(1, numel(this.Storage))];
                if this.Bounds(2) > capacity  %queue has 'wrapped around'. Need to copy the 'wrapped around element to the start of the new storage
                    this.Storage(capacity+1 : this.Bounds(2)) = this.Storage(1 : mod(this.Bounds(2), capacity));
                    this.Storage(1 : mod(this.Bounds(2), capacity)) = {[]}; %and erase them from their previous location
                    %the nice thing about this implementation is that after the moving, this.Bounds(2) now points to the correct location in the new Storage.
                end
                capacity = numel(this.Storage);
                
            end
            this.Storage{mod(this.Bounds(2), capacity) + 1} = in;
            this.Bounds(2) = this.Bounds(2) + 1;
        end
        function out = dequeue(this)
            %DEQUEUE Removes an element from the start of the queue.
            %   out = q.dequeue
            %q: scalar queue object.
            %out: element removed from the queue. This decreases the depth of the queue by one.
            %If the depth of the queue is already 0 (queue is empty), this function errors.
            assert(numel(this) == 1, 'queue:dequeue:nonscalarqueue', 'can only dequeue a scalar queue.');
            assert(this.Depth > 0, 'queue:dequeue:empty', 'Queue is empty.');
            capacity = numel(this.Storage);
            this.Bounds(1) = this.Bounds(1) + 1;
            out = this.Storage{this.Bounds(1)};
            this.Storage{this.Bounds(1)} = [];
            if this.Bounds(1) == capacity
                %this.Bounds(2) is guaranteed to be more than or equal to this.Bounds(1)
                this.Bounds = this.Bounds - capacity;  %so we can decrease both by the capacity
            end
        end
    end
    methods (Static)
        function qarr = MakeVectorOfQueues(n, initialcapacity)
            %MAKEVECTOROFQUEUES Create a row vector of individual queues.
            %   qarr = queue.MakeArray(n)
            %   qarr = queue.MakeArray(n, initialcapacity)
            %n: number of queues in the vector
            %initialcapacity: specifies how many objects the queue can stored initially. Default: 1000.
            %
            %Note that since queue is a handle class, creating an array of queue with the syntax
            %   qarr = repmat(queue, 1, n);
            %results in an array where all the queues are handles of the same queue.
            %Similarly, due to the way matlab initialises object arrays,
            %   qarr(1, n) = queue;  
            %results in an array where all the queues but the last are handles of the same queue.
%             arguments
%                 n (1,1) {mustBeInteger, mustBePositive}
%                 initialcapacity (1,1) {mustBeInteger, mustBePositive} = 1000
%             end
            qarr(1, n) = queue(initialcapacity);  %create queue array. All but the last queue are the same handle due to the way matlab initialise object arrays
            for idx = 1:n-1
                qarr(idx) = queue(initialcapacity);
            end
        end
    end
end            
