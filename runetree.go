package gpt_bpe

import (
	"regexp/syntax"
	"strings"
	"unicode"
)

type RuneNode struct {
	rune              rune               // The rune this node represents.
	runes             []rune             // The prior runes that led to this node.
	terminal          bool               // If this node is an absolute terminal node.
	replacement       *[]rune            // The replacement runes for this node.
	childs            map[rune]*RuneNode // The child nodes.
	childsArr         *[]*RuneNode       // The child nodes in an array, for precedence
	isPrefix          bool               // Whether this node is a valid prefix match
	isContractionTree bool               // Whether this node is a contraction tree
}

type RuneNodes []*RuneNode

func runeIsIn(r rune, runes []rune) bool {
	for _, rr := range runes {
		if r == rr {
			return true
		}
	}
	return false
}

func (nodes *RuneNodes) evaluate(r rune) *RuneNode {
	var idx int
	var candidate *RuneNode
	for idx, candidate = range *nodes {

		var isContraction bool
		if candidate.isContractionTree {
			isContraction = true
		}
		candidate = candidate.evaluate(r)
		// ' is not a contraction but 's is,
		// so we don't care about nils if we're in a contraction tree
		if candidate == nil && isContraction {
			continue
		}

		(*nodes)[idx] = candidate

		if candidate != nil && (candidate.terminal || candidate.
			replacement != nil) {
			break
		}
	}
	// Clean out any nodes that are no longer valid.
	for idx = 0; idx < len(*nodes); idx++ {
		if idx >= len(*nodes) {
			break
		}
		if (*nodes)[idx] == nil {
			*nodes = append((*nodes)[:idx], (*nodes)[idx+1:]...)
			idx--
		}
	}
	return candidate
}

func (node *RuneNode) evaluate(r rune) *RuneNode {
	// If the node has an array of children, use that. The array exists if the
	// node has less than 10 children, and is used to speed up the evaluation
	// of the node.
	if node.childsArr != nil {
		children := *node.childsArr
		for _, child := range children {
			if child.rune == r {
				return child
			}
		}
	} else {
		child, ok := node.childs[r]
		if ok {
			return child
		}
	}
	return nil
}

// Represent the tree as a string by traversing the tree, and using tree
// characters to represent the tree structure.
func (node *RuneNode) string(level int, sb *strings.Builder) {
	if node == nil {
		return
	}
	sb.WriteRune(node.rune)
	idx := 0
	if len(node.childs) == 1 {
		// Get the only element from the map recursively until we find a node
		// with more than one child.
		for r := range node.childs {
			node.childs[r].string(level, sb)
		}
		return
	}
	level += 1
	if node.replacement != nil {
		sb.WriteString(" -> ")
		sb.WriteString(string(*node.replacement))
	}
	sb.WriteByte('\n')

	for r := range node.childs {
		sb.WriteString(strings.Repeat("| ", level-1))
		// If we're the last child, then we prepend with a tree terminator.
		if idx == len(node.childs)-1 {
			sb.WriteString("└─")
		} else {
			sb.WriteString("├─")
		}
		node.childs[r].string(level, sb)
		idx += 1
	}
}

// Wrapper
func (runeTree *RuneNode) String() string {
	sb := strings.Builder{}
	runeTree.string(0, &sb)
	return sb.String()
}

func (runeTree *RuneNode) insertRunes(runes []rune) (node *RuneNode) {
	node = runeTree
	keyLen := len(runes)
	for i := 0; i < keyLen; i++ {
		r := runes[i]
		childNode, ok := node.childs[r]
		if !ok {
			children := make([]*RuneNode, 0)
			node.childs[r] = &RuneNode{
				rune:              r,
				runes:             runes[:i+1],
				terminal:          i == keyLen-1,
				childs:            make(map[rune]*RuneNode, 0),
				childsArr:         &children,
				isContractionTree: node.isContractionTree,
			}
		} else if i == keyLen-1 {
			childNode.terminal = true
		}
		if len(node.childs) > 10 {
			// If there are more than 10 children, we set the array pointer
			// to nil, so that we can use the map instead.
			node.childsArr = nil
		} else {
			if node.childsArr == nil {
				children := make([]*RuneNode, 0)
				node.childsArr = &children
			}
			if len(node.childs) != len(*node.childsArr) {
				*node.childsArr = append(*node.childsArr, node.childs[r])
			}
		}
		node = node.childs[r]
	}
	return node
}

func NewRuneTree() *RuneNode {
	return &RuneNode{
		runes:  []rune{},
		childs: make(map[rune]*RuneNode, 0),
	}
}

// ContractionsTree creates a specialized RuneTree for handling contractions
func CreateContractionsTree() *RuneNode {
	tree := NewRuneTree()
	contractions := []string{
		"'s", "'t", "'re", "'ve", "'m", "'ll", "'d",
	}
	// Insert each contraction into the tree
	for _, c := range contractions {
		tree.insertRunes([]rune(c))
	}
	tree.isContractionTree = true
	return tree
}

func (runeTree *RuneNode) InsertReplacementsIntoRuneTree(
	replacements map[string]string,
) {
	for k, v := range replacements {
		keyRunes := []rune(k)
		valueRunes := []rune(v)
		keyNode := runeTree.insertRunes(keyRunes)
		keyNode.replacement = &valueRunes
	}
}

func CreateReplacementsRuneTree(replacements map[string]string) *RuneNode {
	runeTree := NewRuneTree()
	runeTree.isContractionTree = false
	runeTree.InsertReplacementsIntoRuneTree(replacements)
	return runeTree
}

func (runeTree *RuneNode) InsertIntoRuneTree(s []string) {
	for _, k := range s {
		keyRunes := []rune(k)
		runeTree.insertRunes(keyRunes)
	}
}

// Create a new rune tree from an array of strings to match against.
func CreateRuneTree(s []string) *RuneNode {
	runeTree := NewRuneTree()
	runeTree.isContractionTree = false
	runeTree.InsertIntoRuneTree(s)
	return runeTree
}

type rangeTuple struct {
	start int
	end   int
}

// The AST is given as a []rune where every two runes are the start and end of a range
// We want to convert this to a list of rangeTuples for easier handling
func ArrayAsRanges(runes []rune) []rangeTuple {
	// [65 90 97 122 170 170 181 181 186 186 192 214 216 246 248 705 ...
	// All are pairs of 2, start and end of a range, print as X-Y
	ranges := make([]rangeTuple, 0)
	for i := 0; i < len(runes); i += 2 {
		ranges = append(ranges, rangeTuple{start: int(runes[i]), end: int(runes[i+1])})
	}
	return ranges
}

// We will need to populate a lookup table for the ranges
// Once per node. Use binary search to find the rune in the ranges
func populateCharRanges(i int, ranges []rangeTuple) bool {
	// Binary search
	low, high := 0, len(ranges)-1
	for low <= high {
		mid := low + (high-low)/2
		if ranges[mid].start <= i && i <= ranges[mid].end {
			return true
		}
		if i < ranges[mid].start {
			high = mid - 1
		} else {
			low = mid + 1
		}
	}
	// If we didn't find the rune in the ranges, return false
	return false
}

type RangeLUT struct {
	lookup []bool
}

func newRangeLUT(ranges []rangeTuple) *RangeLUT {
	maxLutSize := ranges[len(ranges)-1].end + 1
	lut := &RangeLUT{
		lookup: make([]bool, maxLutSize),
	}
	for i := 0; i < len(lut.lookup); i++ {
		lut.lookup[i] = populateCharRanges(i, ranges)
	}
	return lut
}

// Once we have done it once, we can now use a lookup table to find the rune in the ranges
func containsCharInRange(r rune, lut *RangeLUT) bool {
	if lut != nil && int(r) < len(lut.lookup) {
		return lut.lookup[int(r)]
	} else {
		return false
	}
}

// Nodes of the regex tree
type RegexNode struct {
	runeArray   []rune       // The runes this node represents, used in literals and char classes
	parent      *RegexNode   // The parent node
	children    []*RegexNode // The child nodes
	min         int          // The min number of matches, set previously, used in literals and char classes
	max         int          // The max number of matches, set previously, used in literals and char classes
	flags       int          // Any flags set on the node, Unused for now
	lastOp      string       // The operation of the node prior
	thisOp      string       // The operation of the node
	pathStrings []string     // The string representation of the path to this node
	rangeLUT    *RangeLUT    // The lookup table for char classes
}

func CreateRegexTree(AST *syntax.Regexp) *RegexNode {
	// Given a syntax.regexp assumed as the root, create a tree of RegexNodes
	// We want the info nodes to inform the op nodes of their min/max, flags, and last op

	// Create the root node
	root := &RegexNode{
		runeArray:   AST.Rune,
		parent:      nil,
		children:    make([]*RegexNode, 0),
		min:         AST.Min,
		max:         AST.Max,
		flags:       int(AST.Flags),
		lastOp:      AST.Op.String(),
		thisOp:      AST.Op.String(),
		pathStrings: make([]string, 0),
	}
	root.parent = root
	root.pathStrings = append(root.pathStrings, "(root)")

	// Create the tree
	ASTPath := make([]string, 0)
	ASTPath = append(ASTPath, "(root)")
	root.createTree(AST, ASTPath)

	return root
}

func (rn *RegexNode) createTree(AST *syntax.Regexp, ASTPath []string) {
	// Create the tree
	lastOp := AST.Op.String()
	ASTPath = append(ASTPath, lastOp)

	for _, sub := range AST.Sub {
		// Create a new node
		newNode := &RegexNode{
			runeArray:   sub.Rune,
			parent:      rn,
			children:    make([]*RegexNode, 0),
			min:         sub.Min,
			max:         sub.Max,
			flags:       int(sub.Flags),
			lastOp:      lastOp,
			thisOp:      sub.Op.String(),
			pathStrings: ASTPath,
		}
		if len(sub.Sub) > 0 {
			newNode.createTree(sub, ASTPath)
		}
		rn.children = append(rn.children, newNode)
	}
}

// We need a path map to know where we are in the tree
func (rn *RegexNode) GeneratePathMap() [][]int {
	var pathMap [][]int
	generatePathMap(rn, 0, []int{}, &pathMap)
	return pathMap
}

func generatePathMap(
	rn *RegexNode,
	parentIndex int,
	currentPath []int,
	pathMap *[][]int,
) {
	// Generate a map of the tree with dfs
	currentPath = append(currentPath, parentIndex)

	// If not already in the map, add the current path
	pathCopy := make([]int, len(currentPath))
	copy(pathCopy, currentPath)
	*pathMap = append(*pathMap, pathCopy)
	for idx, child := range rn.children {
		generatePathMap(child, idx, currentPath, pathMap)
	}

}

func (rn *RegexNode) String() string {
	// Print the tree
	sb := strings.Builder{}
	rn.string(0, &sb)
	return sb.String()
}

func (rn *RegexNode) string(level int, sb *strings.Builder) {
	if rn == nil {
		return
	}
	if len(rn.runeArray) > 50 {
		sb.WriteString(string(rn.runeArray[:50]))
	} else {
		sb.WriteString(string(rn.runeArray))
	}
	idx := 0
	if len(rn.children) == 1 {
		// Get the only element from the map recursively until we find a node
		// with more than one child.
		for r := range rn.children {
			rn.children[r].string(level, sb)
		}
		return
	}
	level += 1
	sb.WriteString(" -> ")
	sb.WriteString(rn.lastOp)
	sb.WriteByte('\n')

	for r := range rn.children {
		sb.WriteString(strings.Repeat("| ", level-1))
		// If we're the last child, then we prepend with a tree terminator.
		if idx == len(rn.children)-1 {
			sb.WriteString("└─")
		} else {
			sb.WriteString("├─")
		}
		rn.children[r].string(level, sb)
		idx += 1
	}
}

// Variables saved during and between traversals
type matchVariables struct {
	matchedWords                []string   // The words that have been matched
	subjectRuneArrIndex         int        // The index of the last rune matched
	subjectRuneCandidateIndices []int      // The indices of the runes that are candidates for matching
	currentNodeIdx              int        // The index of the current node in the path map
	pathMap                     [][]int    // The path map of the tree
	ParentOp                    string     // The operation of the parent node from where we are
	minGroupSize                int        // The minimum number of runes that must be matched
	maxGroupSize                int        // The maximum number of runes that can be matched
	candidateRunes              []rune     // The runes that are candidates for matching
	skipUntilNum                int        // The number of nodes to skip until the next node that isn't a child of the current node
	rootNode                    *RegexNode // The root node of the tree
	endEval                     bool       // Whether we should end the evaluation
	lastInfoOpLevel             int        // The level of the last info op, used for resetting group sizes
	parentMatched               bool       // The direct parent of the current node has at least one match
}

// We want to take a string and use pre-order traversal to match the string to the tree, in a regex-like fashion
// This is much faster than using the regex package.
// The input is a pathmap generate from the regex tree, and the runes to match
// The output is a list of strings that have been matched
func (rn *RegexNode) EvaluateRegexTree(runes []rune, pathMap [][]int) []string {
	// Init variables
	var matchVars matchVariables
	matchVars.matchedWords = make([]string, 0)
	matchVars.subjectRuneArrIndex = 0
	matchVars.currentNodeIdx = 0
	matchVars.minGroupSize = 1
	matchVars.maxGroupSize = -1
	matchVars.candidateRunes = make([]rune, 0, 64)
	matchVars.subjectRuneCandidateIndices = []int{0}
	matchVars.pathMap = pathMap
	matchVars.rootNode = rn
	matchVars.endEval = false
	matchVars.lastInfoOpLevel = 1

	// Start the traversal
	for {
		rn.traverseRegexTree(runes, &matchVars, 0)
		if matchVars.subjectRuneArrIndex >= len(runes) {
			break
		}
		// Reset for next round
		matchVars.currentNodeIdx = 0
		matchVars.minGroupSize = 1
		matchVars.maxGroupSize = -1
		matchVars.candidateRunes = matchVars.candidateRunes[:0]
		matchVars.subjectRuneCandidateIndices[0] = matchVars.subjectRuneArrIndex
		matchVars.subjectRuneCandidateIndices = matchVars.subjectRuneCandidateIndices[:1]
		matchVars.skipUntilNum = 0
		matchVars.endEval = false
		matchVars.lastInfoOpLevel = 1
	}

	return matchVars.matchedWords
}

// The recursive function that traverses the tree
func (rn *RegexNode) traverseRegexTree(
	runes []rune,
	matchVars *matchVariables,
	level int,
) {
	// Pre-order traversal of the tree
	if matchVars.endEval {
		return
	}
	level += 1
	thisNodeMap := matchVars.pathMap[matchVars.currentNodeIdx]
	lastNodeMap := make([]int, 0)
	if matchVars.currentNodeIdx > 0 {
		lastNodeMap = matchVars.pathMap[matchVars.currentNodeIdx-1]
	}
	thisNodeRuneIdx := -1
	thisNodeRuneParentIdx := 0

	// Check if we are at the branch root and have a accumulated split
	if len(thisNodeMap) == 2 && len(matchVars.candidateRunes) != 0 {
		strMatched := string(matchVars.candidateRunes)
		matchVars.matchedWords = append(matchVars.matchedWords, strMatched)
		matchVars.subjectRuneArrIndex += len(matchVars.candidateRunes)

		// Finish Round
		matchVars.endEval = true
		return
	} else if len(thisNodeMap) == 2 {
		// Reset candidate indices if we are bach at the branch root
		matchVars.subjectRuneCandidateIndices[0] = matchVars.subjectRuneArrIndex
		matchVars.subjectRuneCandidateIndices = matchVars.subjectRuneCandidateIndices[:1]
	} else if len(thisNodeMap) != len(lastNodeMap) && len(lastNodeMap) != 0 {
		// We have either traversed up or down the tree
		// Reset parent match variable
		matchVars.parentMatched = false
	}

	// Evaluate the current node
	if matchVars.skipUntilNum == 0 {
		// if the index isn't of the right length, we append the index to the candidate indices
		if len(matchVars.subjectRuneCandidateIndices) < len(thisNodeMap) {
			candidateRuneArray := matchVars.subjectRuneCandidateIndices[len(matchVars.subjectRuneCandidateIndices)-1]
			matchVars.subjectRuneCandidateIndices = append(
				matchVars.subjectRuneCandidateIndices, candidateRuneArray,
			)
		} else {
			// Trim to the right length
			matchVars.subjectRuneCandidateIndices = matchVars.subjectRuneCandidateIndices[:len(thisNodeMap)]
		}
		thisNodeRuneIdx = matchVars.subjectRuneCandidateIndices[len(matchVars.subjectRuneCandidateIndices)-1]
		if len(matchVars.subjectRuneCandidateIndices) > 1 {
			thisNodeRuneParentIdx = matchVars.subjectRuneCandidateIndices[len(matchVars.subjectRuneCandidateIndices)-2]
		}

		switch rn.thisOp {
		case "Alternate":
			// Nothing needs to happen if we have these nodes here
		case "Concat":
			// Nothing needs to happen if we have these nodes here
		case "Quest":
			// Set minmax for the next nodes
			matchVars.minGroupSize = 0
			matchVars.maxGroupSize = 1
			matchVars.lastInfoOpLevel = level
		case "Plus":
			// Set minmax for the next nodes
			matchVars.minGroupSize = 1
			matchVars.maxGroupSize = -1
			matchVars.lastInfoOpLevel = level
		case "Repeat":
			// Set minmax for the next nodes
			matchVars.minGroupSize = rn.min
			matchVars.maxGroupSize = rn.max
			matchVars.lastInfoOpLevel = level
		case "Star":
			// Set minmax for the next nodes
			matchVars.minGroupSize = 0
			matchVars.maxGroupSize = -1
			matchVars.lastInfoOpLevel = level
		case "Literal":
			// Evaluate the literal
			caseInsensitiveFlag := false
			if rn.flags&int(syntax.FoldCase) != 0 {
				caseInsensitiveFlag = true
			}
			matches := 0
			matchArr := make([]rune, 0)
			for i := 0; i < len(rn.runeArray); i++ {
				if thisNodeRuneIdx+i < len(runes) {
					if rn.runeArray[i] == runes[thisNodeRuneIdx+i] {
						matches += 1
						matchArr = append(matchArr, runes[thisNodeRuneIdx+i])
					} else {
						if caseInsensitiveFlag && unicode.IsLetter(rn.runeArray[i]) && unicode.IsLetter(runes[thisNodeRuneIdx+i]) {
							if rn.runeArray[i] == runes[thisNodeRuneIdx+i]+32 {
								matches += 1
								matchArr = append(matchArr, runes[thisNodeRuneIdx+i])
							} else if rn.runeArray[i] == runes[thisNodeRuneIdx+i]-32 {
								matches += 1
								matchArr = append(matchArr, runes[thisNodeRuneIdx+i])
							} else {
								break
							}
						} else {
							break

						}
					}
				}
			}

			// If we are expecting a non-zero match, set the min group size
			// to the length of the rune array (literal	length)
			if matchVars.minGroupSize > 0 {
				matchVars.minGroupSize = len(rn.runeArray)
			}

			// Matches must be at least min group but can exceed max, will be cut off.
			if matchVars.minGroupSize == -1 || matches >= matchVars.minGroupSize {
				if matchVars.maxGroupSize == -1 || matches <= matchVars.maxGroupSize {
					// Matched
					matchVars.parentMatched = true
					if matches != 0 {
						matchVars.candidateRunes = append(
							matchVars.candidateRunes, matchArr...,
						)
						thisNodeRuneIdx += matches
					}
				} else if matches > matchVars.maxGroupSize {
					// Matched, but exceeded max
					// set matches to max
					matches = matchVars.maxGroupSize
					if len(matchArr) > matches {
						matchArr = matchArr[:matches]
					}
					matchVars.candidateRunes = append(
						matchVars.candidateRunes, matchArr...,
					)
					thisNodeRuneIdx += matches
					matchVars.parentMatched = true
				} else {
					// Not matched
					// If the parent is a concat, this is an AND statement, we should skip sibings
					hasConcatParent := false
					for _, path := range rn.pathStrings {
						if path == "Concat" {
							hasConcatParent = true
							break
						}
					}

					// Calculate skip length here
					if hasConcatParent {
						matchVars.skipUntilNum = calcSkipLength(
							matchVars.pathMap, matchVars.currentNodeIdx, true,
						)
						matchVars.candidateRunes = matchVars.candidateRunes[:0]
						// pop one idx
						matchVars.subjectRuneCandidateIndices = matchVars.subjectRuneCandidateIndices[:len(matchVars.subjectRuneCandidateIndices)-1]
					} else {
						matchVars.skipUntilNum = calcSkipLength(
							matchVars.pathMap, matchVars.currentNodeIdx, false,
						)
						// Reset one idx
						matchVars.subjectRuneCandidateIndices[len(matchVars.subjectRuneCandidateIndices)-1] = thisNodeRuneParentIdx
						thisNodeRuneIdx = thisNodeRuneParentIdx
					}
				}
			} else {
				// Not matched
				// If the parent is a concat, this is an AND statement, we should skip sibings
				hasConcatParent := false
				parentPtr := rn.parent
				for {
					if parentPtr == rn {
						break
					}
					if parentPtr.thisOp == "Concat" {
						hasConcatParent = true
						break
					} else if parentPtr.thisOp == "Alternate" {
						break
					} else {
						parentPtr = parentPtr.parent
					}

				}
				// If not matched, we don't care about evaluating the
				// children of the current node (and potentially siblings)
				if hasConcatParent {
					matchVars.skipUntilNum = calcSkipLength(
						matchVars.pathMap, matchVars.currentNodeIdx, true,
					)
					matchVars.candidateRunes = matchVars.candidateRunes[:0]
					// pop one idx
					matchVars.subjectRuneCandidateIndices = matchVars.subjectRuneCandidateIndices[:len(matchVars.subjectRuneCandidateIndices)-1]
				} else {
					matchVars.skipUntilNum = calcSkipLength(
						matchVars.pathMap, matchVars.currentNodeIdx, false,
					)
					// Reset one idx
					matchVars.subjectRuneCandidateIndices[len(matchVars.subjectRuneCandidateIndices)-1] = thisNodeRuneParentIdx
					thisNodeRuneIdx = thisNodeRuneParentIdx
				}
			}
		case "CharClass":
			// Evaluate the char class
			// We generate and use a LUT for the char class as an optimization over directly
			// checking the ranges.
			var lut *RangeLUT
			if rn.rangeLUT == nil {
				rangesArray := ArrayAsRanges(rn.runeArray)
				rn.rangeLUT = newRangeLUT(rangesArray)
			} else {
				lut = rn.rangeLUT
			}

			matches := 0
			for {
				if thisNodeRuneIdx+matches < len(runes) {
					if containsCharInRange(runes[thisNodeRuneIdx+matches], lut) {
						matches += 1
					} else {
						break
					}
				} else {
					break
				}
			}

			// Must be at least min group but can exceed max, will be cut off.
			if matchVars.minGroupSize == -1 || matches >= matchVars.minGroupSize {
				if matchVars.maxGroupSize == -1 || matches <= matchVars.maxGroupSize {
					// Matched
					matchVars.parentMatched = true
					if matches != 0 {
						matchVars.candidateRunes = append(
							matchVars.candidateRunes,
							runes[thisNodeRuneIdx:thisNodeRuneIdx+matches]...,
						)
						thisNodeRuneIdx += matches
					}
				} else if matches > matchVars.maxGroupSize {
					// Matched, but exceeded max
					// set matches to max
					matches = matchVars.maxGroupSize
					matchVars.candidateRunes = append(
						matchVars.candidateRunes,
						runes[thisNodeRuneIdx:thisNodeRuneIdx+matches]...,
					)
					thisNodeRuneIdx += matches
					matchVars.parentMatched = true
				} else {
					// Not matched
					// If the last alt/concat parent was a concat
					hasConcatParent := false
					parentPtr := rn.parent
					for {
						if parentPtr == rn {
							break
						}
						if parentPtr.thisOp == "Concat" {
							hasConcatParent = true
							break
						} else if parentPtr.thisOp == "Alternate" {
							break
						} else {
							parentPtr = parentPtr.parent
						}

					}

					// If not matched, we don't care about evaluating the
					// children of the current node (and potentially siblings)
					if hasConcatParent {
						matchVars.skipUntilNum = calcSkipLength(
							matchVars.pathMap, matchVars.currentNodeIdx, true,
						)
						matchVars.candidateRunes = matchVars.candidateRunes[:0]
						// pop one idx
						matchVars.subjectRuneCandidateIndices = matchVars.subjectRuneCandidateIndices[:len(matchVars.subjectRuneCandidateIndices)-1]
					} else {
						matchVars.skipUntilNum = calcSkipLength(
							matchVars.pathMap, matchVars.currentNodeIdx, false,
						)
						// Reset one idx
						matchVars.subjectRuneCandidateIndices[len(matchVars.subjectRuneCandidateIndices)-1] = thisNodeRuneParentIdx
						thisNodeRuneIdx = thisNodeRuneParentIdx
					}
				}
			} else {
				// Not matched
				// If the parent is a concat, this is an AND statement, we should skip sibings
				hasConcatParent := false
				parentPtr := rn.parent
				for {
					if parentPtr == rn {
						break
					}
					if parentPtr.thisOp == "Concat" {
						hasConcatParent = true
						break
					} else if parentPtr.thisOp == "Alternate" {
						break
					} else {
						parentPtr = parentPtr.parent
					}

				}

				// Calculate skip length here
				if hasConcatParent {
					matchVars.skipUntilNum = calcSkipLength(
						matchVars.pathMap, matchVars.currentNodeIdx, true,
					)
					matchVars.candidateRunes = matchVars.candidateRunes[:0]
					// pop one idx
					matchVars.subjectRuneCandidateIndices = matchVars.subjectRuneCandidateIndices[:len(matchVars.subjectRuneCandidateIndices)-1]
				} else {
					//fmt.Printf("Parent is not concat, skipping children\n")
					matchVars.skipUntilNum = calcSkipLength(
						matchVars.pathMap, matchVars.currentNodeIdx, false,
					)
					// Reset one idx
					matchVars.subjectRuneCandidateIndices[len(matchVars.subjectRuneCandidateIndices)-1] = thisNodeRuneParentIdx
					thisNodeRuneIdx = thisNodeRuneParentIdx
				}
			}

		default:
			// Do nothing if we don't find the operation

		}
	} else {
		// Decrement the skip until num
		matchVars.skipUntilNum -= 1
	}

	// Reset min/max if there is no path to a min/max setting node
	found := false
	if level > matchVars.lastInfoOpLevel {
		matchVars.lastInfoOpLevel = level
	}

	if matchVars.minGroupSize == 1 && matchVars.maxGroupSize == -1 {
		found = true
	} else if matchVars.lastInfoOpLevel != 1 {
		found = true
	}

	if !found {
		matchVars.minGroupSize = 1
		matchVars.maxGroupSize = -1
	}

	// Update the rune candidate idx. If theres not a Alternate,we update the parent
	if thisNodeRuneIdx != -1 {
		parentOp := rn.parent.thisOp
		if parentOp == "Quest" || parentOp == "Plus" || parentOp == "Repeat" || parentOp == "Star" {
			if len(matchVars.subjectRuneCandidateIndices) > 1 {
				matchVars.subjectRuneCandidateIndices[len(matchVars.subjectRuneCandidateIndices)-2] = thisNodeRuneIdx
			}
		}
		matchVars.subjectRuneCandidateIndices[len(matchVars.subjectRuneCandidateIndices)-1] = thisNodeRuneIdx
	}

	// Load info from the current node
	matchVars.currentNodeIdx += 1
	// If next node is a branch root, and this node is a failed match, we want to actively clear the candidate runes
	flagNextNodeIsBranchRoot := false
	if matchVars.currentNodeIdx < len(matchVars.pathMap) && len(matchVars.pathMap[matchVars.currentNodeIdx]) == 2 {
		flagNextNodeIsBranchRoot = true
	}
	if flagNextNodeIsBranchRoot && len(matchVars.candidateRunes) != 0 && !matchVars.parentMatched {
		matchVars.candidateRunes = matchVars.candidateRunes[:0]
	}
	// Traverse the children
	for _, child := range rn.children {
		child.traverseRegexTree(runes, matchVars, level)
	}

}

// Given current index, find the next index that isn't a child of the current index
// If skipSiblings is true, we skip all siblings of the current node as well
// Return the number of nodes between the current node and the next node that isn't a child of the current node
func calcSkipLength(mapOfTree [][]int, currentPos int, skipSiblings bool) int {
	// Get the current path
	currentPath := mapOfTree[currentPos]
	lenOfCurrentPath := len(currentPath)
	skipLength := 0
	for {
		// Check if we are at end of map
		if currentPos == len(mapOfTree)-1 {
			break
		}
		// Check if we are at root
		if len(mapOfTree[currentPos]) == 1 {
			break
		}

		// Siblings are on the same length, if we want to skip siblings, we only check for lesser length
		if skipSiblings {
			if len(mapOfTree[currentPos+1]) < lenOfCurrentPath {
				break
			} else {
				currentPos += 1
			}
		} else {
			if len(mapOfTree[currentPos+1]) <= lenOfCurrentPath {
				break
			} else {
				currentPos += 1
			}
		}

		skipLength += 1
	}
	return skipLength
}
