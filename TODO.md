* generated Pipeline::new() figure out subpass ix automatically
* Migrate stuff to CommandUtil
* expand helpers like CommandUtil to make it easy to implement gfx passes
* sync validation check traversability
* sync validation validate increasing semaphore values
* add automatic code for submitting stuff in the right order (not just validate sync & generate timeline semaphores)
* add validation for transitive dependencies, this is weird sometimes and it should be forbidden to be most concise
