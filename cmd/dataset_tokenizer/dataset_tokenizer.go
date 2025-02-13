package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/endpoints"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/wbrown/gpt_bpe"
	"github.com/wbrown/gpt_bpe/resources"
	"github.com/yargevad/filepathx"
)

type PathInfo struct {
	Path    string
	Size    int64
	ModTime time.Time
	Dir     bool
}

type S3Client interface {
	GetObject(input *s3.GetObjectInput) (*s3.GetObjectOutput, error)
	ListObjectsV2(input *s3.ListObjectsV2Input) (
		*s3.ListObjectsV2Output,
		error,
	)
}

// GlobTexts
// Given a directory path, recursively finds all `.txt`, `.md`, and `.jsonl`
// files, returning a slice of PathInfo.
func GlobTexts(dirPath string) (pathInfos []PathInfo, err error) {
	// If the path is a file, return it.
	if stat, _ := os.Stat(dirPath); stat != nil && !stat.IsDir() {
		return []PathInfo{{
			Path:    dirPath,
			Size:    stat.Size(),
			ModTime: stat.ModTime(),
			Dir:     stat.IsDir(),
		}}, nil
	}

	unfilteredPaths, err := filepathx.Glob(dirPath + "/**/")
	if err != nil {
		return nil, err
	}
	filePaths := make([]string, 0)
	// Filter out non-text files.
	for _, filePath := range unfilteredPaths {
		if strings.HasSuffix(filePath, ".txt") ||
			strings.HasSuffix(filePath, ".md") ||
			strings.HasSuffix(filePath, ".jsonl") {
			filePaths = append(filePaths, filePath)
		}
	}

	numMatches := len(filePaths)
	if numMatches == 0 {
		return nil, fmt.Errorf(
			"%s does not contain any .txt, .md, or .jsonl files",
			dirPath,
		)
	}
	pathInfos = make([]PathInfo, numMatches)
	for matchIdx := range filePaths {
		currPath := filePaths[matchIdx]
		if stat, statErr := os.Stat(currPath); statErr != nil {
			return nil, statErr
		} else {
			pathInfos[matchIdx] = PathInfo{
				Path:    currPath,
				Size:    stat.Size(),
				ModTime: stat.ModTime(),
				Dir:     stat.IsDir(),
			}
		}
	}
	return pathInfos, nil
}

func SortPathInfoBySize(pathInfos []PathInfo, ascending bool) {
	if ascending {
		sort.Slice(
			pathInfos, func(i, j int) bool {
				return pathInfos[i].Size < pathInfos[j].Size
			},
		)
	} else {
		sort.Slice(
			pathInfos, func(i, j int) bool {
				return pathInfos[i].Size > pathInfos[j].Size
			},
		)
	}
}

func SortPathInfoByPath(pathInfos []PathInfo, ascending bool) {
	if ascending {
		sort.Slice(
			pathInfos, func(i, j int) bool {
				return pathInfos[i].Path < pathInfos[j].Path
			},
		)
	} else {
		sort.Slice(
			pathInfos, func(i, j int) bool {
				return pathInfos[i].Path > pathInfos[j].Path
			},
		)
	}
}

func GetFileInfo(paths []string) ([]os.FileInfo, error) {
	fileInfos := make([]os.FileInfo, 0)
	for _, path := range paths {
		if fileInfo, err := os.Stat(path); err != nil {
			return nil, err
		} else {
			fileInfos = append(fileInfos, fileInfo)
		}
	}
	return fileInfos, nil
}

func GetPathsFromFileInfo(fileInfos []os.FileInfo) (paths []string) {
	for _, fileInfo := range fileInfos {
		paths = append(paths, fileInfo.Name())
	}
	return paths
}

func ShufflePathInfos(pathInfos []PathInfo) {
	for i := len(pathInfos) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		pathInfos[i], pathInfos[j] = pathInfos[j], pathInfos[i]
	}
}

// FindNewestPath
// Given a directory path, recursively scans and returns the path and modified
// time for the newest `.txt` file.
func FindNewestPath(paths []PathInfo) (
	path *string, newest *time.Time,
	err error,
) {
	var newestPath string
	var newestTime *time.Time
	for _, pathInfo := range paths {
		if newestTime == nil || newestTime.Before(pathInfo.ModTime) {
			newestTime = &pathInfo.ModTime
			newestPath = pathInfo.Path
		}
	}
	return &newestPath, newestTime, nil
}

// FindNewestText
// Given a directory, recursively scans and returns the path and modified
// time for the newest file.
func FindNewestText(dirPath string) (
	path *string, newest *time.Time,
	err error,
) {
	matches, err := GlobTexts(dirPath)
	if err != nil {
		return nil, nil, err
	}
	return FindNewestPath(matches)
}

// FindNewestDir
// Given a directory, recursively scans and returns the path and modified
// time for the directory that contains the most recent file modification.
func FindNewestDir(dirPath string) (
	path *string, newest *time.Time,
	err error,
) {
	fileMatches, err := GlobTexts(dirPath)
	if err != nil {
		return nil, nil, err
	}
	// Find all directories, as removed files will modify the time of the
	// containing directory.
	directories := make([]PathInfo, 0)
	for matchIdx := range fileMatches {
		if fileMatches[matchIdx].Dir {
			directories = append(directories, fileMatches[matchIdx])
		}
	}
	return FindNewestPath(directories)
}

type namedRuneReader struct {
	path   string
	reader io.RuneReader
}

func resolveSortSpec(matches []PathInfo, sortSpec string) (err error) {
	if sortSpec == "" || sortSpec == "none" {
		return nil
	} else if sortSpec == "shuffle" {
		ShufflePathInfos(matches)
	} else if sortSpec == "size_ascending" {
		SortPathInfoBySize(matches, true)
	} else if sortSpec == "size_descending" {
		SortPathInfoBySize(matches, false)
	} else if sortSpec == "path_ascending" {
		SortPathInfoByPath(matches, true)
	} else if sortSpec == "path_descending" {
		SortPathInfoByPath(matches, false)
	} else {
		return fmt.Errorf("invalid sort spec: %s", sortSpec)
	}
	return nil
}

// getObjectsS3Recursively retrieves objects recursively from an S3 bucket
// and sends them to the objects channel.
func getObjectsS3Recursively(
	svc S3Client,
	bucketName, prefix string,
	objects chan<- *s3.Object,
) {
	var continuationToken *string
	for {
		params := &s3.ListObjectsV2Input{
			Bucket:            aws.String(bucketName),
			Prefix:            aws.String(prefix),
			ContinuationToken: continuationToken,
		}

		resp, err := svc.ListObjectsV2(params)
		if err != nil {
			log.Printf("Error listing objects: %v", err)
			return
		}

		for _, obj := range resp.Contents {
			key := *obj.Key
			if strings.HasSuffix(key, ".txt") || strings.HasSuffix(
				key, ".jsonl",
			) {
				objects <- obj
			}
		}

		if !*resp.IsTruncated {
			break
		}

		continuationToken = resp.NextContinuationToken
	}
}

// fetchJSONLFileS3 reads a JSONL file from S3, extracts the "text" key,
// and return it as a string with spaces.
func fetchJSONLFileS3(svc S3Client, bucketName, objectKey string) (
	string,
	error,
) {
	params := &s3.GetObjectInput{
		Bucket: aws.String(bucketName),
		Key:    aws.String(objectKey),
	}

	resp, err := svc.GetObject(params)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var text strings.Builder
	jsonlReader := bufio.NewReader(resp.Body)

	firstLine := true // Flag to track the first line
	for {
		line, err := jsonlReader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return "", err
		}

		// Parse the JSONL line into a map
		var jsonObjectMap map[string]interface{}
		if err := json.Unmarshal([]byte(line), &jsonObjectMap); err != nil {
			return "", err
		}

		// Extract the "text" field
		textValue, ok := jsonObjectMap["text"].(string)
		if !ok {
			return "",
				fmt.Errorf(
					"JSONL object has no 'text' field or " +
						"it's not a string",
				)
		}

		// Append the text to the result
		if firstLine {
			firstLine = false
		} else {
			text.WriteString(" ") // Append space for lines except first
		}
		text.WriteString(textValue)
	}

	return text.String(), nil
}

// fetchTextFileS3 reads a text file from S3 and return its content as string
func fetchTextFileS3(svc S3Client, bucketName, objectKey string) (
	string,
	error,
) {
	params := &s3.GetObjectInput{
		Bucket: aws.String(bucketName),
		Key:    aws.String(objectKey),
	}

	resp, err := svc.GetObject(params)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var text strings.Builder
	textReader := bufio.NewReaderSize(resp.Body, 8*1024*1024)
	for {
		buf := make([]byte, 4096)
		n, err := textReader.Read(buf)
		if err != nil {
			if err == io.EOF {
				break
			}
			return "", err
		}
		text.Write(buf[:n])
	}

	return text.String(), nil
}

// removeS3Prefix splits the input into the bucket and to ensure that
// s3:// is present
func removeS3Prefix(input string) (
	hasS3Prefix bool,
	remainder string,
	s3FilePath string,
) {
	prefix := "s3://"
	if strings.HasPrefix(input, prefix) {
		//if it is just s3:// then return empty string
		if !strings.Contains(input[len(prefix):], "/") {
			return true, input[len(prefix):], ""
		}
		//if it is s3://bucket/ then return bucket and empty string
		if strings.Index(
			input[len(prefix):], "/",
		) == len(input)-len(prefix)-1 {
			return true, input[len(prefix) : len(input)-1], ""
		}
		//if it is s3://bucket/path then return bucket and path
		if strings.Index(input[len(prefix):], "/") > 0 {
			idxOfFirstSlash := strings.Index(input[len(prefix):], "/")
			bucket := input[len(prefix) : len(prefix)+idxOfFirstSlash]
			pathOfFile := input[len(prefix)+idxOfFirstSlash+1:]
			return true, bucket, pathOfFile
		}
	}
	return false, input, ""
}

// ReadTextsFromS3 reads text files recursively from all prefixes in an S3 bucket.
func ReadTextsFromS3(
	svc S3Client,
	bucketName string,
	s3FilePath string,
	sanitize bool,
	numReaderThreads int,
) (chan namedRuneReader, error) {
	runeReaders := make(chan namedRuneReader, 32)
	objects := make(chan *s3.Object, 64)
	wg := sync.WaitGroup{}

	// Start reader goroutines.
	startReader := func() {
		for {
			object, ok := <-objects
			if !ok {
				break
			}
			if s3FilePath == "" || strings.HasPrefix(
				*object.Key, s3FilePath,
			) {

				if strings.HasSuffix(*object.Key, ".jsonl") {
					// Handle JSONL files.
					jsonObject, err := fetchJSONLFileS3(
						svc, bucketName, *object.Key,
					)

					if err != nil {
						log.Printf(
							"Error reading JSONL file %s: %v", *object.Key,
							err,
						)
						continue
					}

					// Create our rune reader.
					if sanitize {
						runeReaders <- namedRuneReader{
							*object.Key,
							CreateTextSanitizer(strings.NewReader(jsonObject)),
						}
					} else {
						runeReaders <- namedRuneReader{
							*object.Key,
							strings.NewReader(jsonObject),
						}
					}
				} else {
					// Handle regular text files.
					text, err := fetchTextFileS3(svc, bucketName, *object.Key)
					if err != nil {
						log.Printf(
							"Error reading text file %s: %v", *object.Key,
							err,
						)
						continue
					}

					// Create our rune reader.
					if sanitize {
						runeReaders <- namedRuneReader{
							*object.Key,
							CreateTextSanitizer(strings.NewReader(text)),
						}
					} else {
						runeReaders <- namedRuneReader{
							*object.Key,
							strings.NewReader(text),
						}
					}
				}
			}
		}
		wg.Done()
	}

	// Start multiple reader goroutines.
	for i := 0; i < numReaderThreads; i++ {
		wg.Add(1)
		go startReader()
	}

	go func() {
		// List objects recursively.
		getObjectsS3Recursively(svc, bucketName, "", objects)

		// Close the objects channel when done.
		close(objects)

		// Wait for all reader goroutines to finish.
		wg.Wait()

		// Close the runeReaders channel.
		close(runeReaders)
	}()

	return runeReaders, nil
}

func GetJsonObject(jsonlReader *bufio.Reader) (
	map[string]interface{}, error,
) {
	jsonObject, err := jsonlReader.ReadBytes('\n')
	if err != nil {
		return nil, err
	}

	var jsonObjectMap map[string]interface{}
	if err := json.Unmarshal(jsonObject, &jsonObjectMap); err != nil {
		return nil, err
	}

	return jsonObjectMap, nil
}

func ExtractTextsFromJsonObject(jsonObjectMap map[string]interface{}) (
	[]string, error,
) {
	// We do a recursive search for all fields named `text` and extract
	// the text.
	texts := make([]string, 0)
	for key, value := range jsonObjectMap {
		if key == "text" {
			text, ok := value.(string)
			if !ok {
				return nil, errors.New("text field is not a string")
			}
			texts = append(texts, text)
		} else if subMap, ok := value.(map[string]interface{}); ok {
			subTexts, err := ExtractTextsFromJsonObject(subMap)
			if err != nil {
				return nil, err
			}
			texts = append(texts, subTexts...)
		} else if subArray, ok := value.([]interface{}); ok {
			for _, subValue := range subArray {
				if subMap, ok := subValue.(map[string]interface{}); ok {
					subTexts, err := ExtractTextsFromJsonObject(subMap)
					if err != nil {
						return nil, err
					}
					texts = append(texts, subTexts...)
				}
			}
		}
	}
	return texts, nil
}

// ReadTexts
// Consumes a directory path and recursively scans for `.txt` files, producing
// a TextsIterator function that yields the text file as an io.Reader type.
func ReadTexts(
	dirPath string,
	sanitize bool,
	sortSpec string,
	numReaderThreads int,
) (chan namedRuneReader, error) {
	matches, err := GlobTexts(dirPath)
	// If we have a single file and it's not a directory, then we want to
	// adjust the dirPath to the file's directory.
	if len(matches) == 1 && !matches[0].Dir {
		dirPath = filepath.Dir(matches[0].Path)
	}
	if err != nil {
		return nil, err
	}
	if sortErr := resolveSortSpec(matches, sortSpec); sortErr != nil {
		return nil, sortErr
	}

	removeDirPath := func(path string) string {
		noDir := strings.TrimPrefix(path, dirPath)
		if strings.HasPrefix(noDir, "/") {
			return noDir[1:]
		}
		return noDir
	}

	// We pre-emptively do the work to set up the buffers for the next files,
	// while the prior file is being consumed.
	runeReaders := make(chan namedRuneReader, 128)
	paths := make(chan PathInfo, 256)
	wg := sync.WaitGroup{}
	startReader := func() {
		for {
			if path, ok := <-paths; ok {
				if fileReader, openErr := os.Open(path.Path); openErr != nil {
					log.Fatal(openErr)
				} else {
					if strings.HasSuffix(path.Path, ".jsonl") {
						// Split JSONL files into individual JSON objects.
						jsonlReader := bufio.NewReader(fileReader)
						jsonlReader = bufio.NewReaderSize(
							jsonlReader, 32*1024*1024,
						)
						jsonlReader.Peek(1024 * 1024)
						idx := 0
						for {
							jsonObjectMap, jErr := GetJsonObject(jsonlReader)
							if jErr == io.EOF {
								break
							} else if jErr != nil {
								log.Printf(
									"JSONL object %d in %s is not"+
										" valid JSON: %s",
									idx, path.Path, jErr,
								)
								idx++
								continue
							}
							// Extract the text field.
							texts, err := ExtractTextsFromJsonObject(
								jsonObjectMap,
							)
							if err != nil {
								log.Fatal("JSONL object missing text fields")
							}
							// Join the texts with `***\n`
							textString := strings.Join(texts, "***\n")
							// remove the path prefix
							fileName := fmt.Sprintf(
								"%s[%d]",
								removeDirPath(path.Path),
								idx,
							)
							// Create our rune reader.
							if sanitize {
								runeReaders <- namedRuneReader{
									fileName,
									CreateTextSanitizer(
										strings.NewReader(textString),
									)}
							} else {
								runeReaders <- namedRuneReader{
									fileName,
									strings.NewReader(textString)}
							}
							idx++
						}
					} else {
						fileName := removeDirPath(path.Path)
						if sanitize {
							runeReaders <- namedRuneReader{
								fileName,
								CreateTextSanitizer(fileReader)}
						} else {
							bufferedReader := bufio.NewReaderSize(
								fileReader, 8*1024*1024,
							)
							bufferedReader.Peek(1024 * 1024)
							runeReaders <- namedRuneReader{
								fileName,
								bufferedReader}
						}
					}
				}
			} else {
				break
			}
		}
		wg.Done()
	}
	for readerIdx := 0; readerIdx < numReaderThreads; readerIdx++ {
		wg.Add(1)
		go startReader()
	}
	go func() {
		for matchIdx := range matches {
			paths <- matches[matchIdx]
		}
		close(paths)
		wg.Wait()
		close(runeReaders)
	}()
	return runeReaders, nil
}

// TextsTokenizer
// A struct that encapsulates the configuration for a streaming tokenizer.
type TextsTokenizer struct {
	TokenizerId      string
	ContextSize      int
	Boundary         string
	BoundaryBegin    bool
	BoundaryOverlap  int
	PadToken         string
	EndOfText        string
	Unitrim          bool
	ExcludeTokens    []string
	SanitizeEncoding bool
}

// NewTextsTokenizer
// Creates a new TextsTokenizer struct with the default configuration.
func NewTextsTokenizer() TextsTokenizer {
	return TextsTokenizer{
		"gpt2",
		2048,
		"\n",
		false,
		-1,
		"<|endoftext|>",
		"<|padding|>",
		true,
		[]string{},
		false,
	}
}

// getAndCheckToken
// Check if s is a valid token via tokenizer table lookup; if not, we try
// encoding and checking if it is a valid single token returned.
func getAndCheckToken(
	t *gpt_bpe.GPTEncoder, s string,
	id string,
) (gpt_bpe.Token, error) {
	s = strings.ReplaceAll(s, "\\n", "\n")
	token := t.Get(s)
	if token == nil {
		tokens := *t.Encode(&s)
		// Also allow a single "real" token surrounded by an EosToken and/or
		// a BosToken
		if len(tokens) == 1 ||
			len(tokens) == 2 && tokens[1] == t.EosToken && tokens[0] != t.BosToken {
			return tokens[0], nil
		} else if len(tokens) == 3 &&
			tokens[0] == t.BosToken && tokens[2] == t.EosToken ||
			len(tokens) == 2 && tokens[0] == t.BosToken && tokens[1] != t.EosToken {
			return tokens[1], nil
		} else {
			return 0, fmt.Errorf("'%s' is not a valid token for %s", s, id)
		}
	} else {
		return *token, nil
	}
}

func (tt *TextsTokenizer) InitTokenizer() (*gpt_bpe.GPTEncoder, error) {
	var encoderPtr *gpt_bpe.GPTEncoder
	var tokErr error
	embeddedDirName := tt.TokenizerId + "-tokenizer"
	if embedded, _ := resources.EmbeddedDirExists(embeddedDirName); embedded {
		// Check if it's an internal reference. If not, it's a file path.
		encoderPtr, tokErr = gpt_bpe.NewEncoder(embeddedDirName)
	} else {
		// Fall back to path-like.
		encoderPtr, tokErr = gpt_bpe.NewEncoder(tt.TokenizerId)
	}
	if tokErr != nil {
		return nil, tokErr
	}

	if exclErr := tt.handleExclusions(encoderPtr); exclErr != nil {
		return nil, exclErr
	}

	if tt.SanitizeEncoding {
		// Table is found in sanitizer.go
		encoderPtr.SpecialsTree.InsertReplacementsIntoRuneTree(encodingTable)
	}

	return encoderPtr, nil
}

func (tt *TextsTokenizer) PartitionBoundary(
	boundaryIdxes *[]int,
	begin int,
	idx int,
) int {
	if tt.Boundary == "" && tt.BoundaryOverlap >= 0 {
		idx = begin + tt.BoundaryOverlap
	} else if len(*boundaryIdxes) > 0 {
		var boundaryIdx int
		if tt.BoundaryOverlap == -1 {
			boundaryIdx = (*boundaryIdxes)[len(*boundaryIdxes)-1]
		} else {
			// Find the closest boundary token to the index
			// of the BoundaryOverlap.
			smaller := 0
			larger := 0
			for _, boundaryIdx = range *boundaryIdxes {
				if boundaryIdx < begin+tt.BoundaryOverlap {
					smaller = boundaryIdx
				} else {
					larger = boundaryIdx
					break
				}
			}
			if smaller == 0 {
				boundaryIdx = larger
			} else if larger == 0 {
				boundaryIdx = smaller
			} else {
				if tt.BoundaryOverlap-smaller <
					larger-tt.BoundaryOverlap {
					boundaryIdx = smaller
				} else {
					boundaryIdx = larger
				}
			}
		}

		if boundaryIdx > 0 && boundaryIdx <= begin+tt.ContextSize {
			boundaryOffset := 0
			if !tt.BoundaryBegin {
				boundaryOffset = 1
			}
			if idx-boundaryIdx+boundaryOffset <= tt.ContextSize {
				idx = boundaryIdx + boundaryOffset
			}
		}
	}
	return idx
}

func (tt TextsTokenizer) handleExclusions(
	tokenizer *gpt_bpe.GPTEncoder,
) (
	err error,
) {
	if len(tt.ExcludeTokens) == 0 {
		return nil
	}
	// Remove excluded tokens from the tokenizer.
	for _, excludeToken := range tt.ExcludeTokens {
		var excludeTokenId gpt_bpe.Token
		var excludeErr error
		if excludeTokenId, excludeErr = getAndCheckToken(
			tokenizer,
			excludeToken, "ExcludeToken",
		); excludeErr != nil {
			return excludeErr
		} else {
			delete(tokenizer.Encoder, excludeToken)
			// Remove from merges.
			for i, merge := range tokenizer.TokenMerges {
				// Check if the token is in the merge, if it is, remove it
				if merge == excludeTokenId {
					mergePair := gpt_bpe.GPTPair{
						Left:  string(tokenizer.Decoder[i.Left]),
						Right: string(tokenizer.Decoder[i.Right]),
					}
					delete(tokenizer.TokenMerges, i)
					delete(tokenizer.BpeRanks, mergePair)
				}
			}
			// Remove from specials
			for i, special := range tokenizer.Specials {
				if special[0] == excludeTokenId {
					delete(tokenizer.Specials, i)
				}
			}
			tokenizer.UpdateSpecialsTree()
		}
	}
	tokenizer.UpdateSpecialsTree()
	return nil
}

type TokenizerStatus struct {
	Tokenizer        *gpt_bpe.GPTEncoder
	TokenizerState   string
	PartitionerState string
	NumTokens        int
	PadTokens        int
	NumFiles         int
	CurrFile         string
	Texts            *chan namedRuneReader
	Tokenized        *chan gpt_bpe.Tokens
	Contexts         *chan gpt_bpe.Tokens
	ChunkerIndex     int
	AccumulatorSize  int
	TimeTokenizing   time.Duration
	WaitTime         time.Duration
	SendWait         time.Duration
}

func (tt TextsTokenizer) TokenizeTexts(
	texts chan namedRuneReader,
	indexPath string,
	tokenizerPtr *gpt_bpe.GPTEncoder,
) (chan gpt_bpe.Tokens, *TokenizerStatus, error) {
	tokenizerStatus := TokenizerStatus{}
	var tokErr error
	if tokenizerPtr == nil {
		tokenizerPtr, tokErr = tt.InitTokenizer()
		if tokErr != nil {
			return nil, nil, tokErr
		}
	} else {
		tokenizerPtr = tokenizerPtr.Clone()
	}
	tokenizerStatus.Tokenizer = tokenizerPtr
	tokenizer := *tokenizerPtr
	var endOfText gpt_bpe.Token
	if tt.EndOfText == "" {
		endOfText = tokenizer.EosToken
	} else {
		var eotErr error
		if endOfText, eotErr = getAndCheckToken(
			&tokenizer,
			tt.EndOfText, "EndOfText",
		); eotErr != nil {
			return nil, nil, eotErr
		}
	}

	tokenizedTexts := make(chan gpt_bpe.Tokens, 256)

	tokenizerStatus.Tokenizer = &tokenizer
	tokenizerStatus.Tokenized = &tokenizedTexts
	tokenizerStatus.Texts = &texts

	// Our index handle.
	indexFile, iErr := os.Create(indexPath)
	if iErr != nil {
		return nil, nil, iErr
	}

	currOffset := 0
	idxFormat := "{\"path\": \"%s\", \"offset\": %d, \"tokens\": %d}\n"
	nextTokenized := func() {
		for {
			waitBegin := time.Now()
			runeReader, more := <-texts
			numTokens := 0
			if more {
				tokenizerStatus.CurrFile = runeReader.path
				tokenizerStatus.NumFiles += 1
				tokenizerStatus.WaitTime += time.Since(waitBegin)
				beginTokenize := time.Now()
				encodeChunk := tokenizer.StreamingEncode(runeReader.reader)
				for {
					tokenized := encodeChunk(tt.ContextSize * 4)
					if tokenized == nil {
						tokenizedTexts <- gpt_bpe.Tokens{endOfText}
						numTokens += 1
						tokenizerStatus.NumTokens += 1
						tokenizerStatus.TimeTokenizing += time.Since(beginTokenize)
						break
					} else {
						numTokens += len(*tokenized)
						tokenizerStatus.NumTokens += len(*tokenized)
					}
					sendBegin := time.Now()
					tokenizedTexts <- *tokenized
					tokenizerStatus.SendWait += time.Since(sendBegin)
				}
				indexFile.WriteString(
					fmt.Sprintf(
						idxFormat,
						runeReader.path,
						currOffset,
						numTokens,
					),
				)
				currOffset += numTokens
				numTokens = 0
			} else {
				close(tokenizedTexts)
				indexFile.Close()
				break
			}
		}
	}
	go nextTokenized()
	return tokenizedTexts, &tokenizerStatus, nil
}

// TokenizeTextsToContexts
// Consumes a TextsIterator and produces a ContextsIterator iterator function
// that returns tokenized contexts that are fixed and padded out to
// `contextSize`.
func (tt TextsTokenizer) TokenizeTextsToContexts(
	texts chan namedRuneReader,
	tokenizerPtr *gpt_bpe.GPTEncoder,
	outputContexts chan gpt_bpe.Tokens,
	wg *sync.WaitGroup,
) (chan gpt_bpe.Tokens, *TokenizerStatus, error) {
	status := TokenizerStatus{}
	var tokErr error
	if tokenizerPtr == nil {
		tokenizerPtr, tokErr = tt.InitTokenizer()
		if tokErr != nil {
			return nil, nil, tokErr
		}
	}
	tokenizer := *tokenizerPtr.Clone()
	aligner := *tokenizerPtr.Clone()
	status.Tokenizer = &tokenizer
	var padToken, endOfText gpt_bpe.Token
	if tt.PadToken == "" {
		padToken = tokenizer.PadToken
	} else {
		var padErr error
		padToken, padErr = getAndCheckToken(
			&tokenizer, tt.PadToken,
			"PadToken",
		)
		if padErr != nil {
			return nil, nil, padErr
		}
	}
	if tt.EndOfText == "" {
		endOfText = tokenizer.EosToken
	} else {
		var eotErr error
		endOfText, eotErr = getAndCheckToken(
			&tokenizer, tt.EndOfText,
			"EndOfText",
		)
		if eotErr != nil {
			return nil, nil, eotErr
		}
	}

	var boundary gpt_bpe.Token
	if tt.Boundary == "" {
		boundary = 0xFFFFFFFF
	} else {
		var boundaryErr error
		boundary, boundaryErr = getAndCheckToken(
			&tokenizer, tt.Boundary,
			"Boundary",
		)
		if boundaryErr != nil {
			return nil, nil, boundaryErr
		}
	}
	contextSize := tt.ContextSize
	doUnitrim := tt.Unitrim

	var tokens gpt_bpe.Tokens
	var done bool
	var idx, begin int
	boundaryIdxes := make([]int, 0)

	// Consume texts from `nextText()` and tokenize as a `goroutine`.
	tokenizedTexts := make(chan gpt_bpe.Tokens, 16)
	var contexts chan gpt_bpe.Tokens
	if outputContexts != nil {
		contexts = outputContexts
	} else {
		contexts = make(chan gpt_bpe.Tokens, 32)
	}

	nextTokenized := func() {
		for {
			status.TokenizerState = "waiting"
			waitBegin := time.Now()
			runeReader, more := <-texts
			status.WaitTime += time.Since(waitBegin)
			status.TokenizerState = "tokenizing"
			status.CurrFile = runeReader.path
			status.NumFiles += 1
			beginTs := time.Now()
			if more {
				status.WaitTime += time.Since(waitBegin)
				encodeChunk := tokenizer.StreamingEncode(runeReader.reader)
				for {
					tokenized := encodeChunk(contextSize * 4)
					if tokenized == nil {
						duration := time.Since(beginTs)
						tokenizedTexts <- gpt_bpe.Tokens{endOfText}
						status.NumTokens += 1
						status.TimeTokenizing += duration
						break
					}
					status.TokenizerState = "sending"
					tokenizedTexts <- *tokenized
					status.TokenizerState = "sent"
				}
			} else {
				break
			}
		}
		close(tokenizedTexts)
	}
	go nextTokenized()

	// Consumes tokenized texts and resets closured states for token blocks.
	moreTokens := func() {
		moreTokens, more := <-tokenizedTexts
		tokens = append(tokens, moreTokens...)
		if more {
			done = false
		} else {
			done = true
			status.TokenizerState = "done"
		}
	}
	status.Texts = &texts
	status.Tokenized = &tokenizedTexts
	status.Contexts = &contexts

	// Prime the pump by initializing the states.
	moreTokens()

	// Return an iterator function that returns token chunks that are always
	// `contextSize` tokens.
	nextContext := func() *gpt_bpe.Tokens {
		if len(tokens)-idx < contextSize*4 {
			status.PartitionerState = "waiting"
			moreTokens()
			status.PartitionerState = "got_tokens"
		}
		// Loop until we get a full token chunk.
		for {
			numTokens := len(tokens)
			status.AccumulatorSize = numTokens
			status.ChunkerIndex = idx
			status.PartitionerState = "looping"
			if numTokens == 0 {
				status.PartitionerState = "done"
				return nil
			} else if done && idx == numTokens {
				// We're completely done and have no more token chunks to
				// return, so we flush out and pad the last chunk.
				chunk := tokens[begin:]
				padSize := contextSize - len(chunk)
				if padSize > 0 {
					for padIdx := 0; padIdx < padSize; padIdx += 1 {
						chunk = append(chunk, padToken)
					}
					status.PadTokens += padSize
				}
				tokens = tokens[:0]
				idx = 0
				begin = 0
				status.PartitionerState = "done"
				return &chunk
			}
			// Iterate until we reach the end of this text's tokens.
			for idx < numTokens {
				status.PartitionerState = "iterating"
				status.ChunkerIndex = idx
				token := (tokens)[idx]
				// If this is a 'boundary' token, add it to our list.
				status.PartitionerState = "boundary"
				if token == boundary {
					boundaryIdxes = append(boundaryIdxes, idx)
				}
				// Determine if we're at least `contextSize` yet, and if so
				// we do the finalization of this context.
				currWindow := idx - begin
				if currWindow >= contextSize {
					status.PartitionerState = "chunking"
					chunk := (tokens)[begin:]

					if doUnitrim {
						var endAt int
						chunk, endAt = aligner.AlignAndSizeTokens(
							&chunk,
							contextSize,
						)
						idx = begin + endAt
					} else if len(chunk) > contextSize {
						chunk = (tokens)[:contextSize]
					} else {
						idx = begin + len(chunk)
					}

					// If we have less than `contextSize`, we need to pad out
					// the tokens in this context.
					padSize := contextSize - len(chunk)
					if padSize > 0 {
						for padIdx := 0; padIdx < padSize; padIdx += 1 {
							chunk = append(chunk, padToken)
						}
						status.PadTokens += padSize
					}
					// We had one or more boundary tokens in our last context,
					// so depending on the BoundaryOverlap, use the last
					// boundary or the boundary closest to the BounderOverlap
					// index. This effectively copies the chunk from that point
					// on into the next returned context.
					idx = tt.PartitionBoundary(&boundaryIdxes, begin, idx)

					// We were given a hard index to use as the chunk boundary,
					// and it may not be a complete unicode character, so we
					// need to align it to a valid unicode character.
					if boundary == 0xFFFFFFFF && doUnitrim {
						// Ensure that our next chunk is aligned to valid
						// unicode.
						_, offset := tokenizer.AlignAndSizeTokens(
							&chunk,
							idx-begin,
						)
						idx = begin + offset
					}

					boundaryIdxes = boundaryIdxes[:0]

					// Reset the `begin` offsets, move idx, to set up the
					// state for the next invocation of this function.
					if idx > contextSize*6 {
						tokens = tokens[idx:]
						begin = 0
						idx = 0
					} else {
						begin = idx
					}
					numTokens = len(tokens)
					return &chunk
				}
				idx += 1
				status.AccumulatorSize = numTokens
				status.ChunkerIndex = idx
				if len(tokens)-idx < contextSize*2 {
					status.PartitionerState = "fetching"
					moreTokens()
					status.PartitionerState = "got_tokens"
				}
			}
		}
	}

	go func() {
		for {
			context := nextContext()
			if context == nil || len(*context) == 0 {
				status.CurrFile = ""
				break
			} else {
				status.PartitionerState = "sending"
				contexts <- *context
				status.NumTokens += len(*context)
				status.PartitionerState = "sent"
			}
		}
		if outputContexts == nil {
			close(contexts)
		}
		if wg != nil {
			wg.Done()
		}
	}()

	return contexts, &status, nil
}

// GCD, Greatest Common Divisor of two integers.
func GCD(a, b int) int {
	for b != 0 {
		t := b
		b = a % b
		a = t
	}
	return a
}

// WriteContexts
// Consumes a ContextsIterator function and serializes the contexts to an
// aligned binary file.
func WriteContexts(
	outPath string,
	indexPath string,
	contexts chan gpt_bpe.Tokens,
	encoder *gpt_bpe.GPTEncoder,
	sampling int,
	shuffle bool,
	enforceUint32 bool,
	showContexts bool,
) (int, error) {
	totalTokens := 0
	useUint32 := enforceUint32
	// Use uint32 if explicitly requested or if the vocab size is greater
	// than 65536.
	if !useUint32 {
		if encoder == nil {
			return 0, fmt.Errorf(
				"WriteContexts called with " +
					"unknown encoder; cannot determine output byte width",
			)
		} else if len(encoder.Encoder) > 65536 {
			useUint32 = true
			log.Println(
				"warning: tokenizer vocab too large for " +
					"16-bit, outputting as 32-bit",
			)
		}
	}
	if showContexts && encoder == nil {
		showContexts = false
		log.Println("warning: no encoder info, cannot show contexts")
	}

	// We need to create an index file if we are not using streaming encode as well.
	var indexFile *os.File
	var iErr error
	// Our index handle.
	if err := os.MkdirAll(filepath.Dir(indexPath), os.ModePerm); err != nil {
		return 0, err
	}
	indexFile, iErr = os.OpenFile(indexPath, os.O_TRUNC|os.O_RDWR|os.O_CREATE, 0755)
	if iErr != nil {
		return 0, iErr
	}

	// create file AND filepath if not exists
	if err := os.MkdirAll(filepath.Dir(outPath), os.ModePerm); err != nil {
		return 0, err
	}
	outFile, err := os.OpenFile(
		outPath, os.O_TRUNC|os.O_RDWR|os.O_CREATE,
		0755,
	)
	if err != nil {
		return 0, err
	}

	var sampledContexts chan gpt_bpe.Tokens

	if sampling == 100 {
		sampledContexts = contexts
	} else {
		sampledContexts = make(chan gpt_bpe.Tokens, 16)
		go func() {
			samplingIdx := 0
			for context := range contexts {
				// Ignore every `sampling` percent context (rounded to int)
				// If sampling is 80, GCD is 20, LCD is 5, 80% of 5 is 4
				// So we keep 4 and skip 5th
				gcdOfSampling := GCD(sampling, 100)
				lcd := 100 / gcdOfSampling
				samplingFloat := float64(sampling) / 100.0
				skipEveryX := int(math.Round(samplingFloat * float64(lcd)))
				doKeepSampling := sampling == 100 || (samplingIdx%lcd < skipEveryX)
				if doKeepSampling {
					sampledContexts <- context
					if showContexts {
						fmt.Println(len(context))
						fmt.Println("=================================")
						fmt.Println(encoder.Decode(&context))
					}
				}
				samplingIdx += 1
			}
		}()
	}

	endpos := 0
	var buf []byte
	var contextSize int
	var target int64
	var prevTarget int64

	// Sometimes it is requested that we shuffle all contexts as they are
	// written to the file. This is useful for training data, as it can help
	// the model learn better.

	// We keep track of the idxes of the padding we add to the contexts
	type paddingTuple struct {
		contextStart int
		contextEnd   int
		start        int
		end          int
	}
	idxes := make([]paddingTuple, 0)
	idxFormat := "{\"offset\": %d, \"tokens\": %d}\n"
	for context := range sampledContexts {
		binContext, err := context.ToBin(useUint32)
		if err != nil {
			return totalTokens, err
		}
		// We keep track of the final file position
		if endpos == 0 {
			// On the first context, we discern the context size and make the
			// appropriately sized buffer
			contextSize = len(*binContext)
			buf = make([]byte, contextSize)
		}

		// We select a random position in the buffer that is a multiple of
		// the context size
		if endpos == 0 {
			target = 0
		} else {
			contextEnd := endpos / contextSize
			target = int64(rand.Intn(contextEnd)) * int64(contextSize)
		}

		// If shuffling, we store the context found at the target position in
		// the buffer and write it to the end of the file, writing the new
		// context to the target position
		if endpos != 0 && shuffle {
			_, err := outFile.ReadAt(buf, target)
			if err != nil {
				return totalTokens, err
			}

		} else if shuffle {
			// Write the buffer to the end of the file
			if _, err := outFile.Write(*binContext); err != nil {
				return totalTokens, err
			}
		}

		if endpos > 0 && shuffle {
			// Pad to context size if necessary
			if len(*binContext) < contextSize {
				lenBeforePad := len(*binContext)
				for i := len(*binContext); i < contextSize; i++ {
					(*binContext) = append(*binContext, byte(0))
				}
				// Move the paddingTuple to new position if necessary
				for i, idx := range idxes {
					if idx.contextStart == int(prevTarget) &&
						idx.contextEnd == int(prevTarget)+contextSize {
						// Move to end of file
						idxes[i].contextStart = endpos
						idxes[i].contextEnd = endpos + contextSize
					}
				}

				idxes = append(
					idxes, paddingTuple{start: lenBeforePad,
						end: contextSize, contextStart: int(target),
						contextEnd: int(target) + contextSize},
				)
			}
			// Overwrite binContext to location of the context we just read
			_, err := outFile.WriteAt(*binContext, target)
			if err != nil {
				return totalTokens, err
			}

			// Write the context we just read and replaced to the end of the
			// file
			if _, err := outFile.Write(buf); err != nil {
				return totalTokens, err
			}
		} else if !shuffle {
			// Else, we just write the context to the end of the file
			if _, err := outFile.Write(*binContext); err != nil {
				return totalTokens, err
			}
		}

		_, err = indexFile.WriteString(
			fmt.Sprintf(idxFormat, totalTokens, len(context)),
		)

		if err != nil {
			return totalTokens, err
		}

		totalTokens += len(context)
		endpos += len(*binContext)
	}

	// TODO: This is unnecessary, as golang has a Truncate method
	// Write new file with padding removed
	if shuffle {
		newFilePath := strings.Replace(
			outPath,
			".chunk",
			".shuf.chunk",
			1,
		)
		newFile, err := os.OpenFile(
			newFilePath, os.O_TRUNC|os.O_RDWR|os.O_CREATE, 0755,
		)
		defer newFile.Close()
		if err != nil {
			return totalTokens, err
		}
		// Get contexts from original file, skipping idxes
		buf := make([]byte, contextSize)
		outFile.Seek(0, 0)
		currentStartPos := 0
		currentEndPos := currentStartPos + contextSize
		for {
			num, err := outFile.ReadAt(buf, int64(currentStartPos))
			if err != nil {
				if err == io.EOF {
					break
				}
				return totalTokens, err
			}
			if num == 0 {
				break
			}
			thisContext := paddingTuple{contextStart: currentStartPos,
				contextEnd: currentEndPos, start: 0, end: len(buf)}
			// Check if the context is in the idxes
			for _, idx := range idxes {
				if idx.contextStart == thisContext.contextStart &&
					idx.contextEnd == thisContext.contextEnd {
					if idx.start < thisContext.start ||
						thisContext.start == 0 {
						thisContext.start = idx.start
					}
					if idx.end > thisContext.end ||
						thisContext.end == 0 {
						thisContext.end = idx.end
					}
				}
			}
			if thisContext.start != 0 && thisContext.end != 0 {
				newFile.Write(buf[:thisContext.start])

			} else {
				newFile.Write(buf)
			}
			currentStartPos = currentEndPos
			currentEndPos = currentStartPos + contextSize
		}
		// Replace the original file with the new file
		if err := os.Rename(newFilePath, outPath); err != nil {
			return totalTokens, err
		}
	}

	return totalTokens, nil
}

func StatusWatcher(
	tokenizerStatuses []*TokenizerStatus,
) (chan bool, *sync.WaitGroup) {
	done := make(chan bool)
	statusWg := sync.WaitGroup{}
	statusWg.Add(1)
	begin := time.Now()
	collectStatus := func() {
		totalTokens := 0
		totalFiles := 0
		padTokens := 0
		for _, status := range tokenizerStatuses {
			if status != nil {
				totalTokens += status.NumTokens
				totalFiles += status.NumFiles
				padTokens += status.PadTokens
			}
		}
		duration := time.Since(begin).Seconds()
		log.Printf(
			"%d tokens in %0.2fs, %d pad tokens, %0.2f tokens/s, %d docs, avg %0.2f tokens/doc",
			totalTokens, duration, padTokens,
			float64(totalTokens)/duration,
			totalFiles,
			float64(totalTokens)/float64(totalFiles),
		)
		for i, status := range tokenizerStatuses {
			if status != nil {
				currFile := status.CurrFile
				timeTokenizing := status.TimeTokenizing.Round(time.Millisecond)
				sendWait := status.SendWait.Round(time.Millisecond)
				waitTime := status.WaitTime.Round(time.Millisecond)
				internalState := ""
				textsWaiting := len(*status.Texts)
				textsCap := cap(*status.Texts)
				tokenizedWaiting := len(*status.Tokenized)
				tokenizedCap := cap(*status.Tokenized)
				contextsWaiting := 0
				contextsCap := 0
				if status.Contexts != nil {
					contextsWaiting = len(*status.Contexts)
					contextsCap = cap(*status.Contexts)
				}
				lruHitPercent := 0.0
				if status.Tokenizer.LruHits+status.Tokenizer.LruMisses > 0 {
					lruHitPercent = float64(status.Tokenizer.LruHits) /
						float64(status.Tokenizer.LruHits+status.Tokenizer.LruMisses) * 100
				}
				internalState = fmt.Sprintf(
					" [tokenizer=%s, lru=%0.2f%%, texts=%d/%d, tokenized=%d/%d]",
					status.TokenizerState,
					lruHitPercent,
					textsWaiting, textsCap,
					tokenizedWaiting, tokenizedCap,
				)
				if status.Contexts != nil {
					internalState += fmt.Sprintf(
						" [partitioner=%s (idx=%d, acc_sz=%d), contexts=%d/%d]",
						status.PartitionerState,
						status.ChunkerIndex,
						status.AccumulatorSize,
						contextsWaiting, contextsCap,
					)
				}
				log.Printf(
					"  Thread %d:%s %d tokens, %d pad tokens, %d docs, Times: %s tokenizing, %s recv, %s send, File: %s",
					i,
					internalState,
					status.NumTokens,
					status.PadTokens,
					status.NumFiles,
					timeTokenizing,
					waitTime,
					sendWait,
					currFile,
				)

			}
		}
	}
	go func() {
		for {
			select {
			case <-time.After(10 * time.Second):
				collectStatus()
			case <-done:
				collectStatus()
				statusWg.Done()
				return
			}
		}
	}()
	return done, &statusWg
}

func main() {
	tokenizerId := flag.String(
		"tokenizer", "gpt2",
		"tokenizer to use [gpt2, pile, nerdstash_v1, "+
			"nerdstash_v2, llama, llama3, mistral, huggingface-id]",
	)
	contextSize := flag.Int("context", 8192, "context size")
	showContexts := flag.Bool(
		"show_contexts", false,
		"show contexts as they are tokenized",
	)
	endOfText := flag.String(
		"eot", "",
		"end of text token to split texts, can be token or int16 "+
			"token_id",
	)
	padToken := flag.String(
		"pad", "",
		"pad token to pad out contexts, can be <|padding|>, or an "+
			"int16 token_id",
	)
	boundaryToken := flag.String(
		"boundary", "\n",
		"boundary token to split contexts on, can be a string token "+
			"or int16 token_id",
	)
	boundaryBegin := flag.Bool(
		"boundary_begin", false,
		"whether to treat the boundary token as a beginning token for"+
			"a context",
	)
	boundaryOverlap := flag.Int(
		"boundary_overlap", -1,
		"token index in context to approximately overlap contexts on, "+
			"-1 for last boundary token in context",
	)
	outputFile := flag.String(
		"output", "tokenized.chunk",
		"tokenized output file",
	)
	inputDir := flag.String(
		"input", "",
		"input directory",
	)
	unitrimBool := flag.Bool(
		"no_unitrim", false,
		"do not trim contexts to valid unicode",
	)
	forceRetokenization := flag.Bool(
		"retokenize", false,
		"force retokenization even if tokenizer output is newer",
	)
	sanitizeBool := flag.Bool(
		"sanitize", false,
		"sanitize inputs of whitespace issues",
	)
	reorderPaths := flag.String(
		"reorder", "",
		"reorder input files to specification [size_ascending, "+
			"size_descending, name_ascending, name_descending, random, "+
			"shuffle, none]",
	)
	sampling_str := flag.String(
		"sampling", "100", "a integer value from "+
			"0-100 which tells the tokenizer how many chunks to discard in"+
			" %, 60 keeps 60%% chunks",
	)
	streaming_encode := flag.Bool(
		"streaming_encode", false,
		"use streaming encode, which writes to disk as it encodes, "+
			"rather than buffering into contexts",
	)
	numThreads := flag.Int(
		"threads", 4,
		"number of tokenization threads to use",
	)
	numReaderThreads := flag.Int(
		"reader_threads", 2,
		"number of I/O reader threads to use",
	)
	excludeTokens := flag.String(
		"exclude_tokens", "",
		"comma separated list of tokens to exclude from the vocabulary",
	)
	sanitizeEncodingBool := flag.Bool(
		"disable_sanitize_encoding",
		false, "disable sanitizing of misencoding",
	)
	s3Endpoint := flag.String(
		"object_storage_endpoint",
		"https://object.las1.coreweave.com",
		"CW S3 Endpoint to use for fetching data",
	)
	enforceUint32 := flag.Bool(
		"uint32_enforce", false,
		"output tokens as uint32 instead of uint16 (required for "+
			"vocabs with over 2^16 tokens)",
	)

	flag.Parse()
	if *inputDir == "" {
		flag.Usage()
		log.Fatal("Must provide -input for directory source")
	}

	sampling, err := strconv.Atoi(*sampling_str)
	if err != nil {
		log.Fatal("Sampling parameter must be an integer")
	}

	if sampling > 100 || sampling < 0 {
		log.Fatal("Sampling parameter out of the 0-100 bounds")
	}

	log.Printf("Tokenizer definition: %s\n", *tokenizerId)
	log.Printf("Tokenizer input source: %s\n", *inputDir)
	log.Printf("Tokenizer output: %s\n", *outputFile)
	log.Printf("Tokenizer reordering method: %s\n", *reorderPaths)
	log.Printf(
		"Sampling amount (in %s tokens kept): %d%s\n",
		"%", sampling, "%",
	)

	if *reorderPaths != "" {
		if *reorderPaths != "size_ascending" &&
			*reorderPaths != "size_descending" &&
			*reorderPaths != "name_ascending" &&
			*reorderPaths != "name_descending" &&
			*reorderPaths != "random" &&
			*reorderPaths != "shuffle" &&
			*reorderPaths != "none" {
			log.Fatal("Invalid reorder specification")
		}
	}

	var excludeTokensList []string
	if *excludeTokens != "" {
		excludeTokensList = strings.Split(*excludeTokens, ",")
	}

	textsTokenizer := NewTextsTokenizer()
	textsTokenizer.ContextSize = *contextSize
	textsTokenizer.TokenizerId = *tokenizerId
	textsTokenizer.EndOfText = *endOfText
	textsTokenizer.PadToken = *padToken
	textsTokenizer.Boundary = *boundaryToken
	textsTokenizer.BoundaryBegin = *boundaryBegin
	textsTokenizer.BoundaryOverlap = *boundaryOverlap
	textsTokenizer.Unitrim = !*unitrimBool
	textsTokenizer.ExcludeTokens = excludeTokensList
	textsTokenizer.SanitizeEncoding = !*sanitizeEncodingBool

	if !*forceRetokenization {
		if outStat, outErr := os.Stat(*outputFile); !errors.Is(
			outErr,
			os.ErrNotExist,
		) && outErr != nil {
			log.Fatal(outErr)
		} else if errors.Is(outErr, os.ErrNotExist) {
			log.Printf("Creating %s", *outputFile)
		} else if newestPath, newestModTime, newestErr := FindNewestText(
			*inputDir,
		); newestErr != nil {
			log.Fatal(newestErr)
		} else if newestModTime != nil && newestModTime.Before(
			outStat.ModTime(),
		) {
			log.Printf(
				"Newest source `%s` is older than `%s`, "+
					"not retokenizing. "+
					"Use -retokenize to force retokenization.", *newestPath,
				*outputFile,
			)
			os.Exit(1)
		} else if newestDir, newestDirModTime, newestDirErr := FindNewestDir(
			*inputDir,
		); newestDirErr != nil {
			log.Fatal(newestDirErr)
		} else if newestDirModTime != nil && newestDirModTime.Before(
			outStat.ModTime(),
		) {
			log.Printf(
				"Data source directory `%s` has no changes since `%s"+
					"was tokenized. Use -retokenize to force retokenization.",
				*newestDir, *outputFile,
			)
			os.Exit(1)
		}
	}

	encoder, tokErr := textsTokenizer.InitTokenizer()
	if tokErr != nil {
		log.Fatal(tokErr)
	}

	hasS3Prefix, s3Bucket, s3FilePath := removeS3Prefix(*inputDir)

	if hasS3Prefix && *s3Endpoint == "" {
		flag.Usage()
		log.Fatal(
			"Must provide S3 Endpoint if fetching data from CW " +
				"object storage",
		)
	}

	// Declare textReaders
	var textReaders chan namedRuneReader

	if hasS3Prefix && *s3Endpoint != "" {
		defaultResolver := endpoints.DefaultResolver()
		s3CustResolverFn := func(
			service, region string,
			optFns ...func(*endpoints.Options),
		) (endpoints.ResolvedEndpoint, error) {
			if service == "s3" {
				return endpoints.ResolvedEndpoint{
					URL: *s3Endpoint,
				}, nil
			}

			return defaultResolver.EndpointFor(service, region, optFns...)
		}

		awsConfig := aws.Config{
			EndpointResolver: endpoints.ResolverFunc(s3CustResolverFn),
			Region:           aws.String("coreweave-object-storage"),
		}

		sess := session.Must(
			session.NewSessionWithOptions(
				session.Options{
					Config: awsConfig,
				},
			),
		)

		svc := s3.New(sess)
		textReaders, err = ReadTextsFromS3(
			svc, s3Bucket, s3FilePath, *sanitizeBool, *numReaderThreads,
		)

		if err != nil {
			log.Fatal(err)
		}
	} else {
		textReaders, err = ReadTexts(
			*inputDir, *sanitizeBool, *reorderPaths, *numReaderThreads,
		)

		if err != nil {
			log.Fatal(err)
		}
	}

	numTokens := 0
	begin := time.Now()
	var contextTokens int
	if *streaming_encode {
		tokenizerStatuses := make([]*TokenizerStatus, *numThreads)
		wg := sync.WaitGroup{}
		for threadIdx := 0; threadIdx < *numThreads; threadIdx++ {
			wg.Add(1)
			var tokenizerStatus *TokenizerStatus
			go func(threadId int) {
				var contexts chan gpt_bpe.Tokens
				var tokErr error
				indexFilePath := fmt.Sprintf(
					"%s.%d.index",
					*outputFile, threadId,
				)
				outputFilePath := fmt.Sprintf(
					"%s.%d.tokens",
					*outputFile, threadId,
				)
				contexts, tokenizerStatus, tokErr = textsTokenizer.TokenizeTexts(
					textReaders,
					indexFilePath,
					encoder,
				)
				tokenizerStatuses[threadId] = tokenizerStatus
				if tokErr != nil {
					log.Fatal(tokErr)
				}
				total, writeErr := WriteContexts(
					outputFilePath,
					indexFilePath,
					contexts,
					encoder,
					sampling,
					false,
					*enforceUint32,
					*showContexts,
				)
				if writeErr != nil {
					log.Fatal(writeErr)
				}
				contextTokens += total
				wg.Done()
			}(threadIdx)
		}
		done, statusWg := StatusWatcher(tokenizerStatuses)
		// Wait for all threads to finish
		wg.Wait()
		close(done)
		statusWg.Wait()
	} else {
		wg := sync.WaitGroup{}
		tokenizerStatuses := make([]*TokenizerStatus, *numThreads)
		for threadIdx := 0; threadIdx < *numThreads; threadIdx++ {
			var status *TokenizerStatus
			var tokErr error
			wg.Add(1)
			go func(threadId int) {
				contexts := make(chan gpt_bpe.Tokens, 32)
				outputFilePath := fmt.Sprintf(
					"%s.%d.tokens",
					*outputFile, threadId,
				)
				indexFilePath := strings.ReplaceAll(outputFilePath, ".tokens", ".index")
				contexts, status, tokErr = textsTokenizer.TokenizeTextsToContexts(
					textReaders, encoder, contexts, &wg,
				)
				if tokErr != nil {
					log.Fatal(tokErr)
				}
				tokenizerStatuses[threadId] = status
				if tokErr != nil {
					log.Fatal(tokErr)
				}

				total, writeErr := WriteContexts(
					outputFilePath,
					indexFilePath,
					contexts,
					encoder,
					sampling,
					*reorderPaths == "shuffle",
					*enforceUint32,
					*showContexts,
				)
				if writeErr != nil {
					log.Fatal(writeErr)
				}
				contextTokens += total
				wg.Done()
			}(threadIdx)
		}

		done, statusWg := StatusWatcher(tokenizerStatuses)

		// Wait for all threads to finish
		wg.Wait()
		close(done)
		statusWg.Wait()
	}

	duration := time.Since(begin).Seconds()

	log.Printf(
		"%d tokens in %0.2fs, %0.2f tokens/s", numTokens,
		duration, float64(numTokens)/duration,
	)
	log.Printf("Total tokens written: %d", contextTokens)
}
