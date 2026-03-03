---
name: schema-exploration
description: For discovering and understanding database structure, tables, columns, and relationships
---

# Schema Exploration Skill

## When to Use This Skill

Use this skill when you need to:
- Understand the database structure
- Find which tables contain certain types of data
- Discover column names and data types
- Map relationships between tables
- Answer questions like "What tables are available?" or "What columns does the Customer table have?"

## Workflow

### 1. List All Tables
Use `sql_db_list_tables` tool to see all available tables in the database.

### 2. Get Schema for Specific Tables
Use `sql_db_schema` tool with table names to examine:
- **Column names** - What fields are available
- **Data types** - INTEGER, TEXT, DATETIME, etc.
- **Sample data** - 3 example rows to understand content
- **Primary keys** - Unique identifiers for rows
- **Foreign keys** - Relationships to other tables

### 3. Map Relationships
Identify how tables connect:
- Look for columns ending in "Id" (e.g., CustomerId, ArtistId)
- Foreign keys link to primary keys in other tables

### 4. Answer the Question
Provide clear information about:
- Available tables and their purpose
- Column names and what they contain
- How tables relate to each other
- Sample data to illustrate content

## The Chinook Database Schema

The Chinook database has 11 tables representing a digital media store:

1. **Artist** — Music artists (ArtistId, Name)
2. **Album** — Music albums (AlbumId, Title, ArtistId)
3. **Track** — Individual songs (TrackId, Name, AlbumId, GenreId, MediaTypeId, Composer, Milliseconds, Bytes, UnitPrice)
4. **Genre** — Music genres (GenreId, Name)
5. **MediaType** — File formats (MediaTypeId, Name)
6. **Playlist** — User-created playlists (PlaylistId, Name)
7. **PlaylistTrack** — Tracks in playlists (PlaylistId, TrackId)
8. **Customer** — Store customers (CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId)
9. **Employee** — Store employees (EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email)
10. **Invoice** — Customer purchases (InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total)
11. **InvoiceLine** — Items in invoices (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity)

## Key Relationships

```
Artist (ArtistId)
  ↓ 1:many
Album (ArtistId, AlbumId)
  ↓ 1:many
Track (AlbumId, TrackId)
  ↓ 1:many
InvoiceLine (TrackId, UnitPrice, Quantity)
  ↓ many:1
Invoice (InvoiceId, CustomerId, Total)
  ↓ many:1
Customer (CustomerId, Country, SupportRepId)
  ↓ many:1
Employee (EmployeeId)
```

## Tips

- Table names in Chinook are singular and capitalized (Customer, not customers)
- Foreign keys typically have "Id" suffix and match a table name
- Use sample data to understand what values look like
- When unsure which table to use, list all tables first
