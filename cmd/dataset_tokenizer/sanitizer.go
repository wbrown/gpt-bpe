package main

import (
	"bufio"
	"bytes"
	"io"
	"regexp"
	"strings"
)

//var sanitizeTable = map[string]string{
//	"\r":   "",
//	"\n\n": "\n",
//	"\\n":  "\n",
//	" :":   ":",
//	"\t":   " ",
//}

var encodingTable = map[string]string{
	"â‚¬": "€",
	"â€š": "‚",
	"Æ’":  "ƒ",
	"â€ž": "„",
	"â€¦": "…",
	"â€¡": "‡",
	"Ë†":  "ˆ",
	"â€°": "‰",
	"â€¹": "‹",
	"Å’":  "Œ",
	"Å½":  "Ž",
	"â€˜": "‘",
	"â€™": "’",
	"â€œ": "“",
	"â€¢": "•",
	"â€“": "–",
	"â€”": "—",
	"Ëœ":  "˜",
	"â„¢": "™",
	"Å¡":  "š",
	"â€º": "›",
	"Å“":  "œ",
	"Å¾":  "ž",
	"Å¸":  "Ÿ",
	"Â¡":  "¡",
	"Â¢":  "¢",
	"Â£":  "£",
	"Â¤":  "¤",
	"Â¥":  "¥",
	"Â¦":  "¦",
	"Â§":  "§",
	"Â¨":  "¨",
	"Â©":  "©",
	"Âª":  "ª",
	"Â«":  "«",
	"Â®":  "®",
	"Â¯":  "¯",
	"Â°":  "°",
	"Â±":  "±",
	"Â²":  "²",
	"Â³":  "³",
	"Â´":  "´",
	"Âµ":  "µ",
	"Â¶":  "¶",
	"Â·":  "·",
	"Â¸":  "¸",
	"Â¹":  "¹",
	"Âº":  "º",
	"Â»":  "»",
	"Â¼":  "¼",
	"Â½":  "½",
	"Â¾":  "¾",
	"Â¿":  "¿",
	"Ã€":  "À",
	"Ã‚":  "Â",
	"Ãƒ":  "Ã",
	"Ã„":  "Ä",
	"Ã…":  "Å",
	"Ã†":  "Æ",
	"Ã‡":  "Ç",
	"Ãˆ":  "È",
	"Ã‰":  "É",
	"ÃŠ":  "Ê",
	"Ã‹":  "Ë",
	"ÃŒ":  "Ì",
	"ÃŽ":  "Î",
	"Ã‘":  "Ñ",
	"Ã’":  "Ò",
	"Ã“":  "Ó",
	"Ã”":  "Ô",
	"Ã•":  "Õ",
	"Ã–":  "Ö",
	"Ã—":  "×",
	"Ã˜":  "Ø",
	"Ã™":  "Ù",
	"Ãš":  "Ú",
	"Ã›":  "Û",
	"Ãœ":  "Ü",
	"Ãž":  "Þ",
	"ÃŸ":  "ß",
	"Ã¡":  "á",
	"Ã¢":  "â",
	"Ã£":  "ã",
	"Ã¤":  "ä",
	"Ã¥":  "å",
	"Ã¦":  "æ",
	"Ã§":  "ç",
	"Ã¨":  "è",
	"Ã©":  "é",
	"Ãª":  "ê",
	"Ã«":  "ë",
	"Ã¬":  "ì",
	"Ã®":  "î",
	"Ã¯":  "ï",
	"Ã°":  "ð",
	"Ã±":  "ñ",
	"Ã²":  "ò",
	"Ã³":  "ó",
	"Ã´":  "ô",
	"Ãµ":  "õ",
	"Ã¶":  "ö",
	"Ã·":  "÷",
	"Ã¸":  "ø",
	"Ã¹":  "ù",
	"Ãº":  "ú",
	"Ã»":  "û",
	"Ã¼":  "ü",
	"Ã½":  "ý",
	"Ã¾":  "þ",
	"Ã¿":  "ÿ",
}

// SanitizedRuneReader
// SanitizeRuneReader sanitizes the runes from an io.RuneReader, removing
// extra whitespace, and replacing escaped newlines with actual newlines, and
// replacing tabs with spaces.
type SanitizedRuneReader struct {
	bufSize         int
	lastRune        *rune
	whitespaceRegex *regexp.Regexp
	reader          *bufio.Reader
	currBuffer      **bytes.Buffer
	accumulator     *[]rune
	accumulatorIdx  *int
	moreBuffers     chan *bytes.Buffer
}

// nextBuffer accumulates sanitized runes into a buffer, and returns the buffer.
func (runeReader SanitizedRuneReader) nextBuffer() *bytes.Buffer {
	acc := runeReader.accumulator
	accIdx := runeReader.accumulatorIdx
	idx := *accIdx
	var text string
	for {
		if idx > runeReader.bufSize {
			text = string((*acc)[:runeReader.bufSize])
			(*acc)[0] = (*acc)[idx-1]
			*accIdx = 1
			break
		}
		r, size, _ := (*runeReader.reader).ReadRune()
		if size == 0 && idx == 0 {
			// No valid rune, and our accumulator is empty, so we're done.
			return nil
		} else if size == 0 {
			// No valid rune, and we have stuff in our accumulator, so let's
			// flush and finish up.
			text = string((*acc)[:idx])
			*accIdx = 0
			break
		} else if r == '\r' {
			// Silently drop Windows `\r`
		} else if r == '\n' && *runeReader.lastRune == '\n' {
			// Drop additional newlines.
		} else if r == 'n' && *runeReader.lastRune == '\\' {
			// Replace escaped `\n` with `\n`.
			(*acc)[idx-1] = '\n'
		} else if r == ':' && *runeReader.lastRune == ' ' {
			// Strip colons with leading spaces.
			(*acc)[idx-1] = ':'
		} else if r == '\t' {
			// Replace tabs with single spaces.
			(*acc)[idx] = ' '
			idx++
		} else {
			// We have a valid rune, so let's insert it onto our accumulator.
			(*acc)[idx] = r
			idx++
		}
		if idx == 0 {
			*runeReader.lastRune = ' '
		} else {
			*runeReader.lastRune = (*acc)[idx-1]
		}
	}
	lines := strings.Split(text, "\n")
	for lineIdx := range lines {
		line := lines[lineIdx]
		line = runeReader.whitespaceRegex.ReplaceAllString(line, " ")
		line = strings.TrimSpace(line)
		lines[lineIdx] = line
	}
	text = strings.Join(lines, "\n")
	stringBuffer := bytes.NewBufferString(text)
	return stringBuffer
}

// ReadRune implements the bytes.RuneReader interface. It returns the next
// rune from the sanitized reader, along with the size of the rune, and
// whether or not there are any more runes to read.
func (runeReader SanitizedRuneReader) ReadRune() (
	r rune, size int,
	err error,
) {
	if *runeReader.currBuffer == nil {
		return 0, 0, io.EOF
	} else if r, size, err = (*runeReader.currBuffer).ReadRune(); err != nil {
		if newBuffer, ok := <-runeReader.moreBuffers; !ok {
			return 0, 0, io.EOF
		} else {
			*runeReader.currBuffer = newBuffer
			return (*runeReader.currBuffer).ReadRune()
		}
	} else {
		return r, size, err
	}
}

// CreateTextSanitizer creates a SanitizedRuneReader that consumes an
// io.Reader and sanitizes the text it reads. The returned
// SanitizedRuneReader can be used to read scrubbed runes from the io.Reader.
func CreateTextSanitizer(handle io.Reader) SanitizedRuneReader {
	extraWhiteSpace := regexp.MustCompile("[[:space:]]+")
	scanner := bufio.NewReaderSize(handle, 8*1024*1024)
	accumulator := make([]rune, 32769, 32769)
	emptyBuffer := bytes.NewBufferString("")
	lastRune := rune(0)
	accumlatorIdx := 0
	sanitizer := SanitizedRuneReader{
		bufSize:         32768,
		lastRune:        &lastRune,
		whitespaceRegex: extraWhiteSpace,
		reader:          scanner,
		accumulator:     &accumulator,
		accumulatorIdx:  &accumlatorIdx,
		currBuffer:      &emptyBuffer,
		moreBuffers:     make(chan *bytes.Buffer, 1),
	}
	nextBuffer := sanitizer.nextBuffer()
	sanitizer.currBuffer = &nextBuffer
	go func() {
		for {
			if newBuffer := sanitizer.nextBuffer(); newBuffer == nil {
				close(sanitizer.moreBuffers)
				break
			} else {
				sanitizer.moreBuffers <- newBuffer
			}
		}
	}()
	return sanitizer
}

// SanitizeText sanitizes a text file, removing extra whitespace, and
// replacing escaped newlines with actual newlines, and scrubbing colons
// with leading spaces, replacing tabs with spaces, and dropping Windows
// carriage returns.
func SanitizeText(text string) string {
	reader := CreateTextSanitizer(bytes.NewBufferString(text))
	runes := make([]rune, 0)
	for {
		r, size, _ := reader.ReadRune()
		if size > 0 {
			runes = append(runes, r)
		} else {
			break
		}
	}
	return string(runes)
}
